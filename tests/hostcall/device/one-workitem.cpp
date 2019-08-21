#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

__global__ void
kernel(void *buffer, ulong *retval0, ulong *retval1)
{
    uint count = 1;
    ulong arg0 = count++;
    ulong arg1 = count++;
    ulong arg2 = count++;
    ulong arg3 = count++;
    ulong arg4 = count++;
    ulong arg5 = count++;
    ulong arg6 = count++;
    ulong arg7 = count++;

    long2 result;
    result.data = __ockl_hostcall_internal(buffer, SERVICE_TEST, arg0, arg1,
                                           arg2, arg3, arg4, arg5, arg6, arg7);

    *retval0 = result.x;
    *retval1 = result.y;
}

static void
check(hostcall_buffer_t *buffer)
{
    wait_on_signal(buffer->doorbell, 1024 * 1024, SIGNAL_INIT);
    ulong cptr =
        __atomic_load_n(&buffer->ready_stack, std::memory_order_acquire);
    ASSERT(cptr != 0);
    WHEN_DEBUG(std::cout << "received packet: " << std::hex << cptr << std::dec
                         << std::endl);
    ulong fptr =
        __atomic_load_n(&buffer->free_stack, std::memory_order_relaxed);
    WHEN_DEBUG(std::cout << "free stack: " << std::hex << fptr << std::dec
                         << std::endl);
    ASSERT(fptr == 0);

    header_t *header = get_header(buffer, cptr);
    ASSERT(header->next == 0);
    ASSERT(get_ready_flag(header->control) != 0);
    ASSERT(header->activemask == 1);
    ASSERT(header->service == SERVICE_TEST);

    payload_t *payload = get_payload(buffer, cptr);
    auto p = payload->slots[0];

    for (int ii = 0; ii != 8; ++ii) {
        WHEN_DEBUG(std::cout << "payload: " << p[ii] << std::endl);
        ASSERT(p[ii] == ii + 1);
    }
    p[0] = 42;
    p[1] = 17;

    __atomic_store_n(&header->control, reset_ready_flag(header->control),
                     std::memory_order_release);

    // wait for the single wave to return its packet
    ulong F = __atomic_load_n(&buffer->free_stack, std::memory_order_acquire);
    while (F == fptr) {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
        F = __atomic_load_n(&buffer->free_stack, std::memory_order_acquire);
    }
    WHEN_DEBUG(std::cout << "new free stack: " << std::hex << F << std::endl);
    ASSERT(F == inc_ptr_tag(cptr, buffer->index_size));
}

static void
test()
{
    unsigned int numThreads = 1;
    unsigned int numBlocks = 1;

    unsigned int numPackets = 1;

    hsa_signal_t signal;
    ASSERT(hsa_signal_create(SIGNAL_INIT, 0, NULL, &signal) ==
           HSA_STATUS_SUCCESS);

    hostcall_buffer_t *buffer = createBuffer(numPackets, signal);
    ASSERT(buffer);

    void *retval0_void;
    HIPCHECK(hipHostMalloc(&retval0_void, 8));
    uint64_t *retval0 = (uint64_t *)retval0_void;
    *retval0 = 0x23232323;

    void *retval1_void;
    HIPCHECK(hipHostMalloc(&retval1_void, 8));
    uint64_t *retval1 = (uint64_t *)retval1_void;
    *retval1 = 0x17171717;

    hipLaunchKernelGGL(kernel, dim3(numBlocks), dim3(numThreads), 0, 0, buffer,
                       retval0, retval1);
    check(buffer);
    HIPCHECK(hipDeviceSynchronize());
    ASSERT(*retval0 == 42);
    ASSERT(*retval1 == 17);
}

int
main(int argc, char **argv)
{
    hsa_init();
    ASSERT(set_flags(argc, argv));
    test();
    test_passed();
}
