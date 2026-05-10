const DIMS: u32 = 14u;
const PACKED_DIMS: u32 = 8u;
const TOP_K: u32 = 5u;
const WORKGROUP_SIZE: u32 = 256u;
const INVALID_DISTANCE: f32 = 1.0e30;
const INVALID_INDEX: u32 = 4294967295u;

struct Params {
    ref_count: u32,
    refs_per_chunk: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    values0: vec4<i32>,
    values1: vec4<i32>,
    values2: vec4<i32>,
    values3: vec4<i32>,
}

struct Candidate {
    distance: f32,
    index: u32,
    label: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> references0: array<u32>;
@group(0) @binding(1) var<storage, read> references1: array<u32>;
@group(0) @binding(2) var<uniform> query: Query;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> out_candidates: array<Candidate>;

var<workgroup> local_candidates: array<Candidate, 256>;

fn invalid_candidate() -> Candidate {
    return Candidate(INVALID_DISTANCE, INVALID_INDEX, 0u, 0u);
}

fn better(left: Candidate, right: Candidate) -> bool {
    if (left.distance < right.distance) {
        return true;
    }
    if (left.distance > right.distance) {
        return false;
    }
    return left.index < right.index;
}

fn unpack_i16(word: u32, high: bool) -> i32 {
    let shifted = select(word, word >> 16u, high);
    let raw = i32(shifted & 65535u);
    if (raw >= 32768) {
        return raw - 65536;
    }
    return raw;
}

fn reference_value(index: u32, dim: u32) -> i32 {
    let chunk = index / params.refs_per_chunk;
    let local_index = index - (chunk * params.refs_per_chunk);
    let offset = local_index * PACKED_DIMS + (dim >> 1u);
    var word = 0u;

    if (chunk == 0u) {
        word = references0[offset];
    } else if (chunk == 1u) {
        word = references1[offset];
    }

    return unpack_i16(word, (dim & 1u) == 1u);
}

fn query_value(dim: u32) -> i32 {
    if (dim < 4u) {
        return query.values0[dim];
    }
    if (dim < 8u) {
        return query.values1[dim - 4u];
    }
    if (dim < 12u) {
        return query.values2[dim - 8u];
    }
    return query.values3[dim - 12u];
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let ref_index = global_id.x;
    var candidate = invalid_candidate();

    if (ref_index < params.ref_count) {
        var distance = 0.0;
        for (var dim = 0u; dim < DIMS; dim = dim + 1u) {
            let diff = f32(reference_value(ref_index, dim) - query_value(dim));
            distance = distance + diff * diff;
        }
        candidate = Candidate(distance, ref_index, 0u, 0u);
    }

    local_candidates[lid] = candidate;
    workgroupBarrier();

    var k = 2u;
    loop {
        if (k > WORKGROUP_SIZE) {
            break;
        }

        var j = k >> 1u;
        loop {
            if (j == 0u) {
                break;
            }

            let peer = lid ^ j;
            if (peer > lid) {
                let left = local_candidates[lid];
                let right = local_candidates[peer];
                let ascending = (lid & k) == 0u;

                if (ascending) {
                    if (better(right, left)) {
                        local_candidates[lid] = right;
                        local_candidates[peer] = left;
                    }
                } else {
                    if (better(left, right)) {
                        local_candidates[lid] = right;
                        local_candidates[peer] = left;
                    }
                }
            }

            workgroupBarrier();
            j = j >> 1u;
        }

        k = k << 1u;
    }

    if (lid < TOP_K) {
        out_candidates[workgroup_id.x * TOP_K + lid] = local_candidates[lid];
    }
}
