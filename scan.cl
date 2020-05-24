#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global int * input, __global int * output, __local int * a, __local int * b)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    output[gid] = a[lid];
}


__kernel void prop_hillis_steele(__global double *input, __global double *chunk_sum, int array_size) {
    const int global_id = get_global_id(0);
    const int group_id = get_group_id(0);

    if (global_id >= array_size) {
        return;
    }
    input[global_id] += chunk_sum[group_id];
}