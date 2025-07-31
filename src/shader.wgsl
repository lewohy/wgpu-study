// Vertex shader

// vertex shader의 output을 저장할 구조체
struct VertexOutput {
    // gl_Position과 비슷함
    @builtin(position) clip_position: vec4<f32>,
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
    return out;
}

// Fragment shader

// fragment shader의 entry point
// @location(0)은 fragment shader의 첫 번째 출력 색상 채널
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}
