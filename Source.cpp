#include <cstdio>
#include <algorithm>
#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
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

struct FindMaxAVX2_improved : Xbyak::CodeGenerator {
    FindMaxAVX2_improved()
    {
        // calling convention RDI, RSI, RDX, RCX, R8, R9
        // XMM0-7 (ints are passed that way)
        //      RDI - Reference to Result // WIN RCX
        //      RSI - PTR to Array        // WIN RDX
        //      RDX - Num classes         // WIN R8

        // Regsters that need to be preserved: RBX,RBP, R12-R15

        Xbyak::Label mask;
        Xbyak::Label inf;

        mov(r9, r8);
        shr(r9, 3);  // Divide by 8 (eight floats)
        shl(r8, 2);  // num of Output elements * size of float (4)
        shl(r9, 5);  // Trunc to 32 bytes 

        // Compute partial maximums
        vpbroadcastd(ymm0, ptr[rdx]);
        xor_(rax, rax);             // Move offset for next 8 floating point values
        L("for_i");
        cmp(rax, r9); // TODO: move this downwards.
        jz("tail");
        vmovups(ymm1, ptr[rdx + rax]);  // A
        add(rax, 32);                // Move offset for next 8 floating point values
        vmaxps(ymm0, ymm0, ymm1);
        jmp("for_i");
        // Tail execution
        L("tail");
        //sub(r8, r9); // compute number of remaining bytes
        sub(r9, r8);
        jz("horizontal_max");
        add(r9, 32); // consider lea
        lea(r8, ptr[rip + mask]);
        vmovups(ymm2, ptr[r8 + r9]);
        //vbroadcastss(ymm1, ptr[rip + inf]);
        vmaskmovps(ymm1, ymm2, ptr[rdx + rax]);
        vmaxps(ymm0, ymm0, ymm1);
        L("horizontal_max");
        // Get within shortlisted buffer maximum
        //vperm2f128(ymm1, ymm0, ymm0, 1);
        //vmaxps(ymm0, ymm0, ymm1);  //partial maxes in ymm0
        //vpermilps(xmm1, xmm0, 0x1B);
        //vmaxps(ymm0, ymm0, ymm1);  //partial maxes in ymm0
        //vpermilps(xmm1, xmm0, 1);
        //vmaxps(ymm0, ymm0, ymm1);  //ymm0[0:31] contains global maximum
        //vmovss(ptr[rcx], xmm0); // Result <-Max(X[.])
        vextractf128(xmm1, ymm0, 1); // put higher half of ymm0 into xmm1
        vmaxps(xmm0, xmm0, xmm1);
        vmovhlps(xmm1, xmm0); // put higher half of xmm0 into xmm1
        vmaxps(xmm0, xmm1);
        vpshufd(xmm1, xmm0, 0x55); // store xmm[32:64] in xmm1[0:31]
        vmaxps(xmm0, xmm1); //ymm0[0:31] contains global maximum
        vmovss(ptr[rcx], xmm0); // Result <-Max(X[.])
        ret();
        L(mask);
        for (int i = 0; i < 8; i++)
            dd(0x80000000);
        for (int i = 0; i < 8; i++)
            dd(0x00000000);
        L(inf);
        dd(0xff800000);
    }
};

struct FindMaxAVX2_improved2 : Xbyak::CodeGenerator {
    FindMaxAVX2_improved2()
    {
        // calling convention RDI, RSI, RDX, RCX, R8, R9
        // XMM0-7 (ints are passed that way)
        //      RDI - Reference to Result // WIN RCX
        //      RSI - PTR to Array        // WIN RDX
        //      RDX - Num classes         // WIN R8

        // Regsters that need to be preserved: RBX,RBP, R12-R15

        Xbyak::Label mask;
        Xbyak::Label inf;

        mov(r9, rdx);
        shr(r9, 3);  // Divide by 8 (eight floats)
        shl(rdx, 2);  // num of Output elements * size of float (4)
        shl(r9, 5);  // Trunc to 32 bytes 

                     // Compute partial maximums
        vpbroadcastd(ymm0, ptr[rcx]);
        xor_(rax, rax);             // Move offset for next 8 floating point values
        cmp(rax, r9);
        jz("tail");
        L("for_i");
        vmovups(ymm1, ptr[rcx + rax]);  // A
        add(rax, 32);                // Move offset for next 8 floating point values
        vmaxps(ymm0, ymm0, ymm1);
        cmp(rax, r9);
        jnz("for_i");
        // Tail execution
        L("tail");
        sub(rdx, r9);
        cmp(rdx, 16);
        jb("seq");
        vmovups(xmm2, ptr[rcx + rax]);  // A
        add(rax, 16);				// Move offset for next 4 floating point values
        sub(rdx, 16);
        vperm2f128(ymm2, ymm2, ymm2, 0);
        vmaxps(ymm0, ymm0, ymm2);  //partial maxes in ymm0
        L("seq");
        cmp(rdx, 0);
        jz("horizontal_max");
        vpbroadcastd(ymm2, ptr[rcx + rax]);
        vmaxps(ymm0, ymm0, ymm2);  //partial maxes in ymm0
        sub(rdx, 4);
        add(rax, 4);
        jmp("seq");

        //sub(r9, rdx);
        //jz("horizontal_max");
        //add(r9, 32); // consider lea
        //lea(rdx, ptr[rip + mask]);
        //vmovups(ymm2, ptr[rdx + r9]);
        ////vbroadcastss(ymm1, ptr[rip + inf]);
        //vmaskmovps(ymm1, ymm2, ptr[rcx + rax]);
        //vmaxps(ymm0, ymm0, ymm1);
        L("horizontal_max");
        // Get within shortlisted buffer maximum
        vperm2f128(ymm1, ymm0, ymm0, 1);
        vmaxps(ymm0, ymm0, ymm1);  //partial maxes in ymm0
        vpermilps(xmm1, xmm0, 0x1B);
        vmaxps(ymm0, ymm0, ymm1);  //partial maxes in ymm0
        vpermilps(xmm1, xmm0, 1);
        vmaxps(ymm0, ymm0, ymm1);  //ymm0[0:31] contains global maximum
        //vmovss(ptr[rcx], xmm0); // Result <-Max(X[.])
        //vextractf128(xmm1, ymm0, 1); // put higher half of ymm0 into xmm1
        //vmaxps(xmm0, xmm0, xmm1);
        //vmovhlps(xmm1, xmm0); // put higher half of xmm0 into xmm1
        //vmaxps(xmm0, xmm1);
        //vpshufd(xmm1, xmm0, 0x55); // store xmm[32:64] in xmm1[0:31]
        //vmaxps(xmm0, xmm1); //ymm0[0:31] contains global maximum
        //vmovss(ptr[rcx], xmm0); // Result <-Max(X[.])
        ret();
        L(mask);
        for (int i = 0; i < 8; i++)
            dd(0x80000000);
        for (int i = 0; i < 8; i++)
            dd(0x00000000);
        L(inf);
        dd(0xff800000);
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
    FindMaxAVX2_improved find_max;
    auto max_jit = (void (*)(float*, const float*, unsigned))find_max.getCode();

    for (unsigned n = 0; n < batch_size; ++n) {
        const unsigned batch_offset = n * num_classes;
        const float* in_batch_data = &in_data[batch_offset];
        float* out_batch_data = &out_data[batch_offset];
        float max_input;
        max_jit(&max_input, in_batch_data, num_classes);
        for (unsigned c = 0; c < num_classes; ++c) {
            out_batch_data[c] = in_batch_data[c] - max_input;
        }
    }

    //AssertResultsCorrect(out_data, correct_out, batch_size, num_classes);
}

static void ComputeSoftmaxXbyak2(const float* in_data, float* out_data,
    const float* correct_out, unsigned batch_size, unsigned num_classes)
{
    FindMaxAVX2_improved2 find_max;
    auto max_jit = (float(*)(const float*, unsigned))find_max.getCode();

    for (unsigned n = 0; n < batch_size; ++n) {
        const unsigned batch_offset = n * num_classes;
        const float* in_batch_data = &in_data[batch_offset];
        float* out_batch_data = &out_data[batch_offset];
        float max_input;
        max_input = max_jit(in_batch_data, num_classes);
        for (unsigned c = 0; c < num_classes; ++c) {
            out_batch_data[c] = in_batch_data[c] - max_input;
        }
    }

    //AssertResultsCorrect(out_data, correct_out, batch_size, num_classes);
}

// Old softmax
static void ComputeSoftmaxXbyakOld(const float* in_data, float* out_data,
    const float* correct_out, unsigned batch_size, unsigned num_classes)
{
    FindMaxAVX2 find_max;
    auto max_jit = (float(*)(const float*, unsigned))find_max.getCode();

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

static void RunTest(int batch_size, int num_classes, int reps)
{
    // constexpr long long unsigned batch_size = 1;
    // long long unsigned num_classes = 7;
    // constexpr long long unsigned reps = 1000;
    float* in_data = new float[batch_size * num_classes];
    InitData(in_data, batch_size, num_classes);
    float* out_data = new float[batch_size * num_classes];
    float* xybak_out = new float[batch_size * num_classes];
    INIT_PERF();
    MAKE_PERF_VAR();
    for (unsigned i = 0; i < 50; i++)
    {
        ComputeSoftmaxRef(in_data, out_data, batch_size, num_classes);
        const float* correct_out = out_data;
        ComputeSoftmaxXbyak(in_data, xybak_out, correct_out, batch_size, num_classes);
        ComputeSoftmaxXbyak2(in_data, xybak_out, correct_out, batch_size, num_classes);
        ComputeSoftmaxXbyakOld(in_data, xybak_out, correct_out, batch_size, num_classes);
    }

    for (unsigned i = 0; i < reps; i++)
    {
        BEGIN_OVERALL();
        BEGIN();
        ComputeSoftmaxRef(in_data, out_data, batch_size, num_classes);
        END("Ref");
        const float* correct_out = out_data;
        ComputeSoftmaxXbyak(in_data, xybak_out, correct_out, batch_size, num_classes);
        ComputeSoftmaxXbyak2(in_data, xybak_out, correct_out, batch_size, num_classes);
        ComputeSoftmaxXbyakOld(in_data, xybak_out, correct_out, batch_size, num_classes);
        BEGIN();
        ComputeSoftmaxXbyakOld(in_data, xybak_out, correct_out, batch_size, num_classes);
        END("ASM_old");
        BEGIN();
        ComputeSoftmaxXbyak(in_data, xybak_out, correct_out, batch_size, num_classes);
        END("ASM");
        BEGIN();
        ComputeSoftmaxXbyak2(in_data, xybak_out, correct_out, batch_size, num_classes);
        END("ASM2");
        END_OVERALL();
    }

    delete[] in_data;
    delete[] out_data;
    delete[] xybak_out;
}

int main(int argc, char* argv[])
{
    int bs = atoi(argv[1]);
    int num_classes = atoi(argv[2]);
    int repetitions = atoi(argv[3]);
    RunTest(bs, num_classes, repetitions);
    std::cin.get();
    return 0;
}
