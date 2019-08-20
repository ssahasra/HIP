#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
star()
{
    printf("%*d\n", 16, 42);
    printf("%.*d\n", 8, 42);
    printf("%*.*d\n", -16, 8, 42);
    printf("%*.*f %s * %.*s\n", 16, 8, 123.456, "hello", 5, "worldxyz");
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    hipLaunchKernelGGL(star, dim3(1), dim3(1), 0, 0);
    hipStreamSynchronize(0);
    test_passed();
}
