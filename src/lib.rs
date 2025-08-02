use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[repr(C)]
// Pod: Vertex가 Plain Old Data로, 안전한 &[u8]캐스팅 가능함
// Zeroable: Vertex가 모든 필드가 0으로 초기화된 상태로 안전하게 생성될 수 있음을 나타냄
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // vertex의 크기
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            // buffer의 array가 per-vertex인지 per-instance인지 설정
            step_mode: wgpu::VertexStepMode::Vertex,
            // vertex의 field와 1:1로 attribute를 describe함
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

/// 애플리케이션의 상태를 저장할 구조체This will store the state of our game
pub struct State {
    /// 그림을 그릴 window의 part
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// surface가 configure되었는지를 나타내는 플래그
    ///
    /// resize가 발생하면 `true`로 설정됨
    is_surface_configured: bool,
    /// render pipeline
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    window: Arc<Window>,
}

impl State {
    // We don't need this to be async right now,
    // but we will in the next tutorial
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // instance는 GPU를 위한 handle
        // BackendBit::PRIMARY는 Vulkan, Metal, DX12, Browser WebGPU
        // Adapter와 Surface를 만들기 위해 사용함
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        // 실제 그래픽카드를 위한 handle
        // 그래픽카드의 이름이나 adapter가 사용할 backend 등의 정보를 가져올 때 사용할 수 있음
        // Device와 Queue를 생성하기 위해 사용함
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                // rendering backend가 GPU같은 하드웨어가 아닌 소프트웨어를 사용할 지 설정함
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                // 추가적인 feature를 사용하지 않음
                required_features: wgpu::Features::empty(),
                // limits는 생성할 수 있는 특정 자원의 타입 제한을 기술함
                // WebGL은 wgpu의 모든 feature를 지원하지 않으므로 Web으로 빌드시에 몇개를 비활성화해야 함
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // 이 config로 configure는 resize에서 진행함
        let config = wgpu::SurfaceConfiguration {
            // SurfaceTexture가 어떤 용도로 사용될 지 설정함
            // RENDER_ATTACHMENT는 Texture가 화면에 그려지기 위함을 의미함
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // SurfaceTexture가 GPU에 어떻게 저장될 지 설정함
            // SurfaceCapabilities에서 지원하는 format 중 선택한 것을 사용함
            format: surface_format,
            // with, height는 0이면 안됨
            width: size.width,
            height: size.height,
            // 화면과 surface를 어떻게 동기화할 지 설정함
            // FIFO등등을 사용할 수 있는데, 여기서는 사용가능한 첫 번째 옵션을 씀
            present_mode: surface_caps.present_modes[0],
            // transparent window를 위한 무언가라고 함
            // 잘 모르니 일단 첫 번째 옵션을 사용함
            alpha_mode: surface_caps.alpha_modes[0],
            // TextureView를 생성할 때 사용할 TextureFormat의 리스트
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                // buffer의 descriptor를 설정함
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // color data를 surface에 저장하려면 fragment가 필요함
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // 어떤 color output을 사용할 지 설정
                targets: &[Some(wgpu::ColorTargetState {
                    // surface와 같은 format을 사용하여 복사가 쉬워짐
                    format: config.format,
                    // old pixel을 교체하도록 설정
                    blend: Some(wgpu::BlendState::REPLACE),
                    // 모든 color를 write하도록 설정
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            // vertices를 기본 도형으로 변환할 때 어떻게 해석할지 설정
            primitive: wgpu::PrimitiveState {
                // 3개의 점을 삼각형으로 해석
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // 어느 방향이 앞면인지 설정
                front_face: wgpu::FrontFace::Ccw,
                // back방향 면은 렌더링에 포함하지 않음
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                // 몇 개의 pipeline을 사용할지 설정
                // multisampling을 사용하지 않으므로 1로 설정
                count: 1,
                // 어떤 sample mask를 사용할지 설정
                // 모든 bit를 1로 설정하여 모든 sample이 활성화됨
                mask: !0,
                // antialiasing을 사용하지 않으므로 false로 설정
                alpha_to_coverage_enabled: false,
            },
            // render attachment가 얼마나 많은 array layers를 가질지 설정
            // array texture로 렌더링하지 않으므로 None으로 설정
            multiview: None,
            // shader compilation을 캐싱할지 설정함. 안드로이드 타겟에서나 효과적임
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let num_vertices = VERTICES.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            window,
        })
    }

    /// 윈도우가 resize되는 경우
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            // config를 State::new에서 저장해뒀기 때문에 값만 변경하고 configure만 다시하면 됨
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    /// 렌더링
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        // surface가 configure되지 않았다면 아무것도 하지 않음
        if !self.is_surface_configured {
            return Ok(());
        }

        // surface가 렌더링할 SurfaceTexture를 제공해 줄 때 까지 기다림
        let output = self.surface.get_current_texture()?;

        // 기본 설정으로 TextureView를 생성함
        // render code가 어떻게 texture와 상호작용할지 컨트롤하기를 원하므로 필요함
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // command를 GPU에 보내기 위해 command encoder를 생성함
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            // Render Pass를 생성함
            // _render_pass가 drop되어야 encoder의 대여가 끝나므로 새 스코프 안에서 작업함
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                // color을 어디에 그릴지 설정
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // color를 저장한 texutre를 설정
                    view: &view,
                    // resolved output을 받을 texture
                    // multisampling이 사용되지 않으면 view와 같음
                    resolve_target: None,
                    // screen에서 무엇을 할지 설정
                    ops: wgpu::Operations {
                        // 이전 frame에서 저장된 color를 어떻게 처리할지 설정
                        // 여기서는 화면을 clear할거임
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        // rendered result를 texture에 저장할지 설정
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // render pass에 pipeline을 설정
            render_pass.set_pipeline(&self.render_pipeline);
            // vertex buffer 설정
            render_pass.set_vertex_buffer(
                // 사용할 vertex buffer에 대한 slot
                0,
                self.vertex_buffer.slice(..),
            );
            render_pass.draw(
                // vertex buffer의 vertex갯수만큼 그림
                0..self.num_vertices,
                // 1번 그림
                0..1,
            );
        }

        // command buffer를 종료하고 GPU에 제출
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// key이벤트 처리
    pub fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            // escape 키가 눌렸을 때 exit
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    /// 업데이트 처리
    fn update(&mut self) {
        // remove `todo!()`
    }
}

pub struct App {
    // proxy는 Web에서만 필요함. wgpu resource를 생성하는 과정이 async이기 때문
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,

    // State::new()는 window를 필요로 하고 window는 Resumed State가 되기 전까지 생성할 수 없으므로 Option
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

/// 키 입력, 마우스 입력, lifecycle 이벤트 등을 처리하는 핸들러
impl ApplicationHandler<State> for App {
    /// Web specific한 attribute를 정의함
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on web we can use pollster to
            // await the
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy
                        .send_event(
                            State::new(window)
                                .await
                                .expect("Unable to create canvas!!!")
                        )
                        .is_ok())
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    // panic hook을 설정하여 Web에서 panic 발생 시 콘솔에 출력하도록 함
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
