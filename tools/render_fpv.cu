// Render Rust-generated camera poses using CUDA. Frames are packed RGB24 on stdout.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int texture_size = 256;
constexpr int max_gates = 16;

struct Gate {
    float3 center, normal, side;
};
struct Pose {
    float3 position, right, up, forward;
};
struct Scene {
    int width, height, gate_count;
    float lens[8];
    float3 minimum, maximum;
    float3 bar_min[4], bar_max[4];
    Gate gates[max_gates];
};

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

std::istream& read(std::istream& file, float3& v) {
    return file >> v.x >> v.y >> v.z;
}

__device__ float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 scale(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 sub(float3 a, float3 b) {
    return add(a, scale(b, -1.f));
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 basis(const Gate& gate, float3 v) {
    return make_float3(dot(gate.normal, v), dot(gate.side, v), v.z);
}

__device__ float nonzero(float x) {
    return fabsf(x) < 1e-10f ? 1e-10f : x;
}

__device__ float slab(float3 p, float3 dir, float3 lo, float3 hi, float limit) {
    const float ax = (lo.x-p.x)/nonzero(dir.x), bx = (hi.x-p.x)/nonzero(dir.x);
    const float ay = (lo.y-p.y)/nonzero(dir.y), by = (hi.y-p.y)/nonzero(dir.y);
    const float az = (lo.z-p.z)/nonzero(dir.z), bz = (hi.z-p.z)/nonzero(dir.z);
    const float entry = fmaxf(0.015f, fmaxf(fminf(ax,bx), fmaxf(fminf(ay,by), fminf(az,bz))));
    const float exit = fminf(limit, fminf(fmaxf(ax,bx), fminf(fmaxf(ay,by), fmaxf(az,bz))));
    return entry <= exit ? entry : -1.f;
}

__device__ float distance_to_wall(float position, float direction, float first, float last) {
    return direction > 0.f ? (last-position)/direction : (first-position)/fminf(direction,-1e-12f);
}

__device__ bool near_grid(float value, float spacing, float tolerance) {
    return fabsf(value - roundf(value/spacing)*spacing) < tolerance;
}

__device__ float3 trace(const Scene& s, const Pose& p, const uchar3* texture, float x, float y) {
    const float u = (x/s.width - s.lens[2])/s.lens[0];
    const float v = (s.lens[3] - y/s.height)/s.lens[1];
    const float radius = sqrtf(u*u + v*v);
    const float k1 = s.lens[4], k2 = s.lens[5], k3 = s.lens[6], k4 = s.lens[7];
    float angle = radius;
    for (int it=0; it<5; ++it) {
        const float t = angle*angle;
        const float poly = 1.f + t*(k1+t*(k2+t*(k3+t*k4)));
        const float slope = 1.f + t*(3.f*k1+t*(5.f*k2+t*(7.f*k3+t*9.f*k4)));
        angle -= (angle*poly-radius)/slope;
    }
    const float k = radius > 0.f ? sinf(angle)/radius : 1.f;
    const float3 dir = add(add(scale(p.right, u*k), scale(p.up, v*k)), scale(p.forward, cosf(angle)));
    const float tx = distance_to_wall(p.position.x, dir.x, s.minimum.x, s.maximum.x);
    const float ty = distance_to_wall(p.position.y, dir.y, s.minimum.y, s.maximum.y);
    const float tz = distance_to_wall(p.position.z, dir.z, s.minimum.z, s.maximum.z);
    float distance = fminf(tx, fminf(ty, tz));
    const float3 hit = add(p.position, scale(dir, distance));
    float3 color;
    if (distance == tz && dir.z < 0.f) {
        const int fiber = (int(floorf(hit.x*151.7f)) ^ int(floorf(hit.y*171.9f))) & 15;
        const float level = 158.f + cosf(hit.x*1.1f)*4.f + cosf(hit.y*0.8f)*4.f + fiber*0.53f;
        color = make_float3(level+2.f, level+1.f, level+2.f);
    } else if (distance == tz) {
        const float level = 80.f + cosf(hit.x*1.3f)*16.f;
        color = near_grid(hit.x,4.8f,0.22f) && near_grid(hit.y,4.65f,0.16f)
            ? make_float3(253.f,249.f,224.f) : make_float3(level+9.f,level+10.f,level);
    } else {
        const float along = distance == tx ? hit.y : hit.x;
        const float fold = cosf(along*10.f + sinf(hit.z*0.72f)*0.7f)*9.f;
        const float spill = (cosf(along*0.62f)+1.f)*(3.f+hit.z*1.4f);
        const float level = 36.f + fold + spill;
        color = make_float3(level+1.f,level+1.f,level-6.f);
    }

    for (int i = 0; i < s.gate_count; ++i) {
        const Gate gate = s.gates[i];
        const float3 delta = sub(gate.center, p.position);
        if (dot(delta, dir) < sqrtf(fmaxf(0.f, dot(delta,delta)-4.f))) continue;
        const float3 origin = basis(gate, scale(delta,-1.f)), local = basis(gate, dir);
        for (int j = 0; j < 4; ++j) {
            const float t = slab(origin,local,s.bar_min[j],s.bar_max[j],distance);
            if (t < 0.f) continue;
            const float3 contact = add(origin, scale(local,t));
            if (fabsf(contact.x) >= 0.395f) {
                const float side = contact.x > 0.f ? contact.y : -contact.y;
                const int col = int(fminf(255.f,fmaxf(0.f,(side/2.7f+0.5f)*256.f)));
                const int row = int(fminf(255.f,fmaxf(0.f,(0.5f-contact.z/2.7f)*256.f)));
                const uchar3 texel = texture[row*texture_size+col];
                color = make_float3(texel.x,texel.y,texel.z);
            } else {
                color = make_float3(53.f,30.f,93.f);
            }
            distance = t;
        }
    }
    const float fog = 1.f / (1.f + 0.005f*fmaxf(0.f,distance));
    return add(scale(color,fog),scale(make_float3(27.f,25.f,34.f),1.f-fog));
}

__device__ uint32_t hash(uint32_t n) {
    n ^= n >> 16;
    n *= 0x7feb352du;
    n ^= n >> 15;
    n *= 0x846ca68bu;
    return n ^ (n >> 16);
}

__device__ uchar3 breakup(uchar3 pixel, int x, int y, int width, uint32_t frame, float strength) {
    constexpr uint32_t seed = 1;
    const uint32_t phase = hash(seed ^ (frame/8));
    const float noise = strength*(0.15f+0.85f*float(phase&255u)/255.f);
    const uint32_t cluster = hash(phase ^ (uint32_t(x/13)*0x8da6b343u) ^ (uint32_t(y/3)*0xd8163841u));
    const float intensity = (cluster&255u) < uint32_t(10.f*noise) ? 0.78f : 0.035f;
    const uint32_t sample = hash(hash(seed ^ (frame*0x9e3779b9u)) ^ uint32_t(y*width+x));
    if ((sample&65535u) >= uint32_t(65535.f*intensity*noise)) return pixel;
    switch ((sample>>16)&7u) {
        case 0: case 1: return make_uchar3(15,14,21);
        case 2: case 3: return make_uchar3(235,241,232);
        case 4: return make_uchar3(195,54,196);
        case 5: return make_uchar3(53,226,104);
        default: return make_uchar3(112,121,134);
    }
}

__global__ void render(Scene scene, Pose pose, const uchar3* texture, uchar3* out, uint32_t frame, float strength) {
    const int x = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
    const int y = int(blockIdx.y)*int(blockDim.y)+int(threadIdx.y);
    if (x >= scene.width || y >= scene.height) return;
    float3 color = make_float3(0.f,0.f,0.f);
    for (int row=0; row<2; ++row) {
        for (int col=0; col<2; ++col) {
            color = add(color,trace(scene,pose,texture,x+0.25f+0.5f*col,y+0.25f+0.5f*row));
        }
    }
    const uchar3 pixel = make_uchar3(uint8_t(color.x*0.25f),uint8_t(color.y*0.25f),uint8_t(color.z*0.25f));
    out[y*scene.width+x] = strength > 0.f ? breakup(pixel,x,y,scene.width,frame,strength) : pixel;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("Usage: render_fpv POSES GATE_RGB [HDZERO_STRENGTH]");
        const float strength = argc == 4 ? std::strtof(argv[3],nullptr) : 0.f;
        if (!(strength >= 0.f && strength <= 1.f)) throw std::runtime_error("Strength must be in [0,1]");
        std::ifstream poses(argv[1]), input(argv[2], std::ios::binary);
        Scene scene{};
        int fps = 0;
        if (!(poses >> scene.width >> scene.height >> fps >> scene.gate_count) ||
            scene.width < 1 || scene.width > 4096 || scene.height < 1 || scene.height > 4096 ||
            scene.gate_count < 1 || scene.gate_count > max_gates) {
            throw std::runtime_error("Invalid pose file header");
        }
        for (float& parameter : scene.lens) poses >> parameter;
        read(poses,scene.minimum); read(poses,scene.maximum);
        for (int i=0; i<4; ++i) { read(poses,scene.bar_min[i]); read(poses,scene.bar_max[i]); }
        for (int i=0; i<scene.gate_count; ++i) {
            read(poses,scene.gates[i].center); read(poses,scene.gates[i].normal); read(poses,scene.gates[i].side);
        }
        std::array<uchar3,texture_size*texture_size> tex;
        if (!input.read(reinterpret_cast<char*>(tex.data()),sizeof(tex)) || !poses) {
            throw std::runtime_error("Could not read scene or gate texture");
        }
        std::vector<uchar3> frame(size_t(scene.width)*scene.height);
        uchar3 *gpu_texture, *gpu_frame;
        check(cudaMalloc(&gpu_texture,sizeof(tex)));
        check(cudaMalloc(&gpu_frame,frame.size()*sizeof(uchar3)));
        check(cudaMemcpy(gpu_texture,tex.data(),sizeof(tex),cudaMemcpyHostToDevice));
        const dim3 block(16,16);
        const dim3 grid((scene.width+15)/16,(scene.height+15)/16);
        Pose pose{};
        size_t frames = 0;
        while (read(poses,pose.position)) {
            if (!read(poses,pose.right) || !read(poses,pose.up) || !read(poses,pose.forward)) {
                throw std::runtime_error("Incomplete camera pose");
            }
            render<<<grid,block>>>(scene,pose,gpu_texture,gpu_frame,static_cast<uint32_t>(frames),strength);
            check(cudaGetLastError());
            check(cudaMemcpy(frame.data(),gpu_frame,frame.size()*sizeof(uchar3),cudaMemcpyDeviceToHost));
            if (std::fwrite(frame.data(),sizeof(uchar3),frame.size(),stdout) != frame.size()) {
                throw std::runtime_error("RGB pipe write failed");
            }
            ++frames;
        }
        check(cudaFree(gpu_frame)); check(cudaFree(gpu_texture));
        std::fprintf(stderr,"CUDA rendered %zu frames at %dx%d (%d fps output)\n",frames,scene.width,scene.height,fps);
        return frames == 0 ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what());
        return 1;
    }
}
