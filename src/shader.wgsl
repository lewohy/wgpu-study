// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    // gl_Position과 비슷함
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// vertex shader의 entry point
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);

    return out;
}

// Fragment shader

// fragment shader의 entry point
// @location(0)은 fragment shader의 첫 번째 출력 색상 채널
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
