#include <cstdio>
#include <algorithm>
#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include <iomanip>
#include <map>
#include <intrin.h>

#define INIT_PERF() RtdscHelper rtdsc_helper
#define MAKE_PERF_VAR() unsigned long long perf = 0; (void) perf
#define BEGIN() perf = __rdtsc()
#define END(name) rtdsc_helper.AddMeasurement(name, __rdtsc() - perf)
#define BEGIN_OVERALL() unsigned long long overall = __rdtsc()
#define END_OVERALL() rtdsc_helper.AddMeasurement("Overall", __rdtsc() - overall)

class RtdscHelper
{
    using uint64 = unsigned long long;
public:
    void AddMeasurement(std::string name, uint64 time) {
        if (m_Measurements.find(name) != m_Measurements.end()) {
            m_Measurements[name].first += time;
            m_Measurements[name].second++;
        }
        else
            m_Measurements[name] = { time, 1 };
    }

    void PrintResults() {
        std::cout << "Measurement results" << std::endl;
        auto width = std::setw(20);
        std::cout << std::left << width << "Name"
            << width << "Avg Time"
            << width << "Ratio" << std::endl;
        auto overall_m = m_Measurements["Overall"];
        auto overall = overall_m.first / (double)overall_m.second;
        for (auto const& m : m_Measurements) {
            auto average = m.second.first / (double)m.second.second;
            std::cout << std::left << width << m.first
                << width << average / 2600000000
                << width << average / overall << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    ~RtdscHelper() {
        PrintResults();
    }
private:
    std::map<std::string, std::pair<uint64, unsigned>> m_Measurements; // name, time, count
};

struct FindMaxAVX2 : Xbyak::CodeGenerator {
    FindMaxAVX2()
    {
        // Assumptions:
        // - iterations = num_classes
        // - iterations are assumed to be >= 1
        // LIN:RDI WIN:RCX = input_data
        // LIN:RSI WIN:RDX = iterations
        // LIN:RAX WIN:RAX = iterator
        vmovups(ymm0, ptr[rcx]);
        cmp(edx, 1);
        jbe("horizontal_max");
        lea(rax, ptr[rcx + 32]);
        lea(rdx, ptr[rcx + rdx * 4]);
        L("elementwise_max");
        vmovups(ymm1, ptr[rax]);
        vmaxps(ymm0, ymm0, ymm1);
        add(rax, 32);
        cmp(rax, rdx);
        jbe("elementwise_max");
        L("horizontal_max");
        vextractf128(xmm1, ymm0, 1);
        vmaxps(xmm0, xmm0, xmm1);
        vmovhlps(xmm1, xmm0);
        vmaxps(xmm0, xmm1);
        vpshufd(xmm1, xmm0, 85);
        vmaxps(xmm0, xmm1);
        ret();
    }
};

static void AssertResultsCorrect(const float* data, const float* correct,
    unsigned batch_size, unsigned num_classes)
{
    for(unsigned n = 0; n < batch_size; n++)
    {
        for(unsigned c = 0; c < num_classes; c++)
        {
            _ASSERT(data[n*num_classes + c] == correct[n*num_classes + c]);
        }
    }
}

static void ComputeSoftmaxRef(const float* in_data, float* out_data, unsigned batch_size, unsigned num_classes)
{
    for (unsigned n = 0; n < batch_size; ++n) {
        const unsigned batch_offset = n * num_classes;
        const float* in_batch_data = &in_data[batch_offset];
        float* out_batch_data = &out_data[batch_offset];
        float current_max = in_batch_data[0];
        for (unsigned c = 1; c < num_classes; ++c) {
            current_max = max(in_batch_data[c], current_max);
        }
        for (unsigned c = 0; c < num_classes; ++c) {
            out_batch_data[c] = in_batch_data[c] - current_max;
        }
    }
}

static void ComputeSoftmaxXbyak(const float* in_data, float* out_data,
    const float* correct_out, unsigned batch_size, unsigned num_classes)
{
    FindMaxAVX2 find_max;
    auto max_jit = (float (*)(const float*, unsigned))find_max.getCode();

    for (unsigned n = 0; n < batch_size; ++n) {
        const unsigned batch_offset = n * num_classes;
        const float* in_batch_data = &in_data[batch_offset];
        float* out_batch_data = &out_data[batch_offset];
        const unsigned divisible_num_classes = num_classes & ~7;
        float max_input = max_jit(in_batch_data, divisible_num_classes);
        for (unsigned c = divisible_num_classes; c < num_classes; ++c) {
            max_input = max(in_batch_data[c], max_input);
        }
        for (unsigned c = 0; c < num_classes; ++c) {
            out_batch_data[c] = in_batch_data[c] - max_input;
        }
    }

    //AssertResultsCorrect(out_data, correct_out, batch_size, num_classes);
}

static void InitData(float* in_data, unsigned batch_size, unsigned num_classes)
{
    srand(1);
    for(unsigned n = 0; n < batch_size; n++)
    {
        for(unsigned c = 0; c < num_classes; c++)
        {
            in_data[n*num_classes + c] = rand();
        }
    }
}

static void RunTest()
{
    constexpr long long unsigned batch_size = 1;
    constexpr long long unsigned num_classes = 8 * 1024 * 1024;
    float* in_data = new float[batch_size * num_classes];
    InitData(in_data, batch_size, num_classes);
    float* out_data = new float[batch_size * num_classes];
    float* xybak_out = new float[batch_size * num_classes];
    INIT_PERF();
    MAKE_PERF_VAR();
    for (unsigned i = 0; i < 300; i++)
    {
        BEGIN_OVERALL();
        BEGIN();
        ComputeSoftmaxRef(in_data, out_data, batch_size, num_classes);
        END("Ref");
        const float* correct_out = out_data;
        BEGIN();
        ComputeSoftmaxXbyak(in_data, xybak_out, correct_out, batch_size, num_classes);
        END("ASM");
        END_OVERALL();
    }

    delete[] in_data;
    delete[] out_data;
    delete[] xybak_out;
}

int main()
{
    RunTest();
    std::cin.get();
    return 0;
}
