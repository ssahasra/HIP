#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
specifiers()
{
    const char *N = nullptr;

    printf("xyzzy\n");
    printf("%%\n");
    printf("hello %% world\n");
    printf("%%s\n");
    // Two special tests to make sure that the compiler pass correctly
    // skips over a '%%' without affecting the logic for locating
    // string arguments.
    printf("%%s%p\n", (void*)0xf01dab1eca55e77e);
    printf("%%c%s\n", "xyzzy");
    printf("%c%c%c\n", 's', 'e', 'p');
    printf("%d\n", -42);
    printf("%u\n", 42);
    printf("%f\n", 123.456);
    printf("%F\n", -123.456);
    printf("%e\n", -123.456);
    printf("%E\n", 123.456);
    printf("%g\n", 123.456);
    printf("%G\n", -123.456);
    printf("%c\n", 'x');
    printf("%s\n", N);
    printf("%p\n", N);
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    hipLaunchKernelGGL(specifiers, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(0);
    test_passed();
}
