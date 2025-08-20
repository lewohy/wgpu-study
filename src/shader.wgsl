// Vertex shader

// vertex shader의 output을 저장할 구조체
struct VertexOutput {
    // gl_Position과 비슷함
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pipeline_index: u32,
};

// vertex shader의 entry point
@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.pipeline_index = 0u;
    return out;
}

@vertex
fn vs_main_challenge(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.pipeline_index = 1u;
    return out;
}

// Fragment shader

// fragment shader의 entry point
// @location(0)은 fragment shader의 첫 번째 출력 색상 채널
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.pipeline_index == 0u {
        return vec4<f32>(0.3, 0.2, 0.1, 1.0);
    }
    let pi = 3.141592f;

    let x = in.clip_position.x;
    let y = in.clip_position.y;


    return vec4<f32>(cos(x / 50f), sin(y / 50f), sin(x / 50f), 1.0);
}
