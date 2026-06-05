import gradio as gr
import numpy as np

from perceptron_core import (
    ACTIVATIONS,
    init_weights,
    predict_scores,
    scores_to_percentages,
    train_weights,
    weights_to_images,
)

VERSION = "2.0.0"
AUTHOR_URL = "https://github.com/kairavb"
BIAS = 100
MIN_CLASSES, MAX_CLASSES, DEFAULT_CLASSES = 2, 10, 4
IMAGE_SIZES = [64, 128, 256, 512]
STREAM_EVERY = 0.3
UNTRAINED_MSG = "Train the model to see live classification."

CSS = """
.main-title { margin-bottom: 0.25rem !important; }
.subtitle { margin-top: 0; opacity: 0.75; font-size: 0.95rem; }
.left-panel, .right-panel { gap: 0.5rem !important; }
.compact-webcam, .compact-webcam .wrap { max-height: 240px !important; }
.compact-webcam video, .compact-webcam img { max-height: 220px !important; object-fit: contain !important; }
.capture-grid { flex-wrap: wrap !important; gap: 0.35rem !important; }
.capture-grid button { min-width: 3.75rem !important; padding: 0.45rem 0.85rem !important; }
.weights-gallery { min-height: 300px !important; }
"""


def label(i: int) -> str:
    return f"C{i + 1}"


def make_state(n_classes, resize, color, activation, captures=None):
    return {
        "n_classes": int(n_classes),
        "captures": captures or [[] for _ in range(n_classes)],
        "resize": int(resize),
        "color": bool(color),
        "activation": str(activation),
        "trained": False,
    }


def get_weights(state):
    if "weights" not in state or state["weights"] is None:
        state["weights"] = init_weights(state["resize"], state["color"], state["n_classes"])
    return state["weights"]


def summary(state):
    return " | ".join(f"{label(i)}: {len(state['captures'][i])}" for i in range(state["n_classes"]))


def btn_vis(n_classes):
    return [gr.update(visible=i < n_classes) for i in range(MAX_CLASSES)]


def ui_outputs(state):
    n = state["n_classes"]
    return state, summary(state), f"{n} classes", UNTRAINED_MSG, *btn_vis(n)


def apply_settings(size_str, color_mode, activation, state):
    resize, color = int(size_str), color_mode == "Color"
    return ui_outputs(make_state(state["n_classes"], resize, color, activation))


def change_classes(state, delta, size_str, color_mode, activation):
    new_n = state["n_classes"] + delta
    if not MIN_CLASSES <= new_n <= MAX_CLASSES:
        return ui_outputs(state)
    resize, color = int(size_str), color_mode == "Color"
    captures = [list(state["captures"][i]) if i < len(state["captures"]) else [] for i in range(new_n)]
    return ui_outputs(make_state(new_n, resize, color, activation, captures))


def on_frame(state, frame):
    if frame is None:
        return None, gr.update()
    if not state["trained"]:
        return frame, gr.update()
    act = ACTIVATIONS[state["activation"]]
    raw = predict_scores(frame, get_weights(state), act, state["resize"], state["color"])
    pcts = scores_to_percentages(raw, state["activation"])
    best = max(range(len(pcts)), key=lambda i: pcts[i])
    lines = [f"{label(i)}: {'█' * int(p / 10)}{'░' * (10 - int(p / 10))} {p:.1f}%" for i, p in enumerate(pcts)]
    lines.append(f"→ {label(best)} ({pcts[best]:.1f}%)")
    return frame, "\n".join(lines)


def capture(state, image, class_id):
    if image is not None:
        state["captures"][class_id].append(image.copy())
    return state, summary(state)


def train(state):
    samples = [(i, img) for i in range(state["n_classes"]) for img in state["captures"][i]]
    if not samples:
        return state, "Capture images first.", summary(state)
    act = ACTIVATIONS[state["activation"]]
    weights = get_weights(state)
    state["weights"] = train_weights(samples, weights, BIAS, act, state["resize"], state["color"])
    state["trained"] = True
    state["captures"] = [[] for _ in range(state["n_classes"])]
    return state, f"Done — trained on {len(samples)} images.", summary(state)


def show_weights(state):
    if not state["trained"]:
        return []
    return [(img, label(i)) for i, img in enumerate(weights_to_images(get_weights(state), state["resize"], state["color"]))]


def build_ui():
    initial = make_state(DEFAULT_CLASSES, 256, True, "relu")

    with gr.Blocks(title="Single Layer Perceptron") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        state = gr.State(initial)
        latest_frame = gr.State(value=None)

        gr.Markdown(f"# Single Layer Perceptron v{VERSION} · [by kairav]({AUTHOR_URL})", elem_classes=["main-title"])
        gr.Markdown("Webcam capture → train → live classify. Pure NumPy.", elem_classes=["subtitle"])

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["left-panel"]):
                webcam = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam", height=240, elem_classes=["compact-webcam"])
                classify = gr.Textbox(label="Live classification", value=UNTRAINED_MSG, interactive=False, lines=6)
                restart_btn = gr.Button("Restart program", variant="secondary")
                weights_btn = gr.Button("Show weights")

            with gr.Column(scale=5, elem_classes=["right-panel"]):
                with gr.Group():
                    gr.Markdown("### Settings")
                    with gr.Row():
                        activation_dd = gr.Dropdown(["relu", "sigmoid"], value="relu", label="Activation")
                        color_dd = gr.Dropdown(["Color", "Black & White"], value="Color", label="Mode")
                        size_dd = gr.Dropdown([str(s) for s in IMAGE_SIZES], value="256", label="Size")

                with gr.Group():
                    gr.Markdown("### Classes & capture")
                    with gr.Row():
                        remove_btn = gr.Button("−", scale=1, min_width=40)
                        class_count = gr.Textbox(label="Count", value=f"{DEFAULT_CLASSES} classes", interactive=False, scale=2)
                        add_btn = gr.Button("+", scale=1, min_width=40)
                    counts = gr.Textbox(label="Captured", value=summary(initial), interactive=False)
                    capture_btns = []
                    with gr.Row(elem_classes=["capture-grid"]):
                        for i in range(MAX_CLASSES):
                            capture_btns.append(gr.Button(label(i), visible=i < DEFAULT_CLASSES, min_width=64, scale=0))

                with gr.Group():
                    gr.Markdown("### Train")
                    train_btn = gr.Button("Train model", variant="primary")
                    train_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Learned weights")
        weights_gallery = gr.Gallery(label="Weights", columns=MAX_CLASSES, height=300, object_fit="contain", elem_classes=["weights-gallery"])

        webcam.stream(
            on_frame,
            inputs=[state, webcam],
            outputs=[latest_frame, classify],
            stream_every=STREAM_EVERY,
            time_limit=3600,
        )
        restart_btn.click(None, None, None, js="() => { window.location.reload(); }")

        settings_in = [size_dd, color_dd, activation_dd, state]
        class_in = [state, size_dd, color_dd, activation_dd]
        refresh_out = [state, counts, class_count, classify, *capture_btns]

        add_btn.click(lambda s, sz, c, a: change_classes(s, 1, sz, c, a), class_in, refresh_out)
        remove_btn.click(lambda s, sz, c, a: change_classes(s, -1, sz, c, a), class_in, refresh_out)
        for dd in (activation_dd, color_dd, size_dd):
            dd.change(apply_settings, settings_in, refresh_out)

        for i, btn in enumerate(capture_btns):
            btn.click(lambda s, img, cid=i: capture(s, img, cid), [state, latest_frame], [state, counts])

        train_btn.click(train, [state], [state, train_status, counts])
        weights_btn.click(show_weights, [state], [weights_gallery])

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
