"""
draw_repsegnet_v2_diagram.py
Generates an exact, publication-ready architectural diagram of RepSegNet V2
strictly matching repsegnet_v2.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    fig = plt.figure(figsize=(18, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Color Palette (Publication Grade)
    c_input    = "#E0F2FE" # Light Blue
    c_encoder  = "#BAE6FD" # Sky Blue
    c_bottleneck = "#7DD3FC" # Blue
    c_scse     = "#FED7AA" # Soft Orange / Amber
    c_decoder  = "#BBF7D0" # Soft Mint / Green
    c_aux      = "#FDE68A" # Warm Yellow
    c_head     = "#DDD6FE" # Light Purple
    c_post     = "#FECDD3" # Soft Rose
    c_text     = "#0F172A" # Dark Slate
    c_border   = "#334155" # Slate Border

    # Helper: Draw rounded box
    def draw_box(x, y, w, h, bg, title, subtitle="", edge=c_border, lw=1.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=bg, edgecolor=edge, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=c_text, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                    fontsize=8, color='#334155', style='italic', zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=c_text, zorder=3)
        return box

    # Title Banner
    ax.text(9, 10.5, "RepSegNet V2 Implementation Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=c_text)
    ax.text(9, 10.15, "Pretrained EfficientNet Backbone + Concurrent sCSE Attention Skip Pathways + Deep Supervision",
            ha='center', va='center', fontsize=10.5, color='#475569', style='italic')

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MACRO ENCODER-DECODER PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    # Input Image
    draw_box(0.5, 8.4, 2.0, 0.9, c_input, "Input Image", "3 × 512 × 512")

    # Encoder Stages (Left Column: Downwards)
    enc_info = [
        ("Encoder Stage 1", "24 × 256 × 256", 8.4),
        ("Encoder Stage 2", "32 × 128 × 128", 6.9),
        ("Encoder Stage 3", "48 × 64 × 64", 5.4),
        ("Encoder Stage 4", "136 × 32 × 32", 3.9),
        ("Encoder Stage 5", "384 × 16 × 16", 2.4),
    ]

    # Arrow from Input to Enc 1
    ax.annotate("", xy=(3.0, 8.85), xytext=(2.5, 8.85),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    for idx, (title, dims, y_pos) in enumerate(enc_info):
        draw_box(3.0, y_pos, 2.4, 0.9, c_encoder, title, dims)
        if idx < 4:
            # Downward arrows between encoder stages
            ax.annotate("", xy=(4.2, y_pos - 0.6), xytext=(4.2, y_pos),
                        arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))
            ax.text(4.45, y_pos - 0.3, "Stride 2", fontsize=7.5, color='#64748B')

    # Bottleneck (Bottom Center)
    draw_box(3.0, 0.7, 2.4, 1.0, c_bottleneck, "Bottleneck", "Conv-BN-ReLU × 2\n256 × 16 × 16")
    ax.annotate("", xy=(4.2, 1.7), xytext=(4.2, 2.4),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # Arrow from Bottleneck to Decoder Block 0
    ax.annotate("", xy=(6.5, 1.2), xytext=(5.4, 1.2),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # Decoder Stages (Right Column: Upwards)
    dec_info = [
        ("Decoder Block 0", "256 × 16 × 16", 0.7),
        ("Decoder Block 1", "128 × 32 × 32", 2.4),
        ("Decoder Block 2", "64 × 64 × 64", 3.9),
        ("Decoder Block 3", "32 × 128 × 128", 5.4),
        ("Decoder Block 4", "16 × 256 × 256", 6.9),
    ]

    for idx, (title, dims, y_pos) in enumerate(dec_info):
        draw_box(6.5, y_pos, 2.5, 0.95, c_decoder, title, dims)
        if idx > 0:
            # Upward arrows between decoders
            prev_y = dec_info[idx-1][2]
            ax.annotate("", xy=(7.75, y_pos), xytext=(7.75, prev_y + 0.95),
                        arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))
            ax.text(7.95, prev_y + 0.5, "ConvTrans 2x2", fontsize=7, color='#64748B')

    # Skip Connections with Concurrent sCSE Attention
    # Connect Enc 5 -> Dec 0, Enc 4 -> Dec 1, Enc 3 -> Dec 2, Enc 2 -> Dec 3, Enc 1 -> Dec 4
    skip_mappings = [
        (4, 0, 2.4, 0.7, "sCSE (384)"),
        (3, 1, 3.9, 2.4, "sCSE (136)"),
        (2, 2, 5.4, 3.9, "sCSE (48)"),
        (1, 3, 6.9, 5.4, "sCSE (32)"),
        (0, 4, 8.4, 6.9, "sCSE (24)"),
    ]

    for enc_idx, dec_idx, y_enc, y_dec, scse_label in skip_mappings:
        # Draw small sCSE block on skip line
        scse_x = 5.6
        scse_y = (y_enc + y_dec) / 2 if enc_idx == 4 else y_enc
        
        # Horizontal skip line
        ax.plot([5.4, 5.6], [y_enc + 0.45, y_enc + 0.45], color='#D97706', lw=1.5, linestyle='--')
        
        # sCSE box
        scse_box = patches.FancyBboxPatch(
            (5.55, y_enc + 0.2), 0.75, 0.5,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=c_scse, edgecolor='#D97706', linewidth=1.2, zorder=4
        )
        ax.add_patch(scse_box)
        ax.text(5.92, y_enc + 0.45, "sCSE", ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#9A3412', zorder=5)

        # Path from sCSE to Decoder
        if y_enc == y_dec + 0.45 or abs(y_enc - y_dec) < 0.6:
            ax.annotate("", xy=(6.5, y_dec + 0.45), xytext=(6.3, y_enc + 0.45),
                        arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.5, linestyle='--'))
        else:
            # Step path
            ax.plot([6.3, 6.4, 6.4], [y_enc + 0.45, y_enc + 0.45, y_dec + 0.6],
                    color='#D97706', lw=1.5, linestyle='--')
            ax.annotate("", xy=(6.5, y_dec + 0.6), xytext=(6.4, y_dec + 0.6),
                        arrowprops=dict(arrowstyle="->", color='#D97706', lw=1.5, linestyle='--'))

    # Final Stage: 1x1 Conv Head & Bilinear Interpolation
    draw_box(6.5, 8.4, 2.5, 0.9, c_head, "Final Conv Head", "Conv 1×1 + Sigmoid\n1 × 256 × 256")
    ax.annotate("", xy=(7.75, 8.4), xytext=(7.75, 7.85),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # Resize / Interpolation to 512x512
    draw_box(9.6, 8.4, 2.0, 0.9, c_head, "Bilinear Resize", "1 × 512 × 512")
    ax.annotate("", xy=(9.6, 8.85), xytext=(9.0, 8.85),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # Connected-Component Post-Processing
    draw_box(12.1, 8.4, 2.5, 0.9, c_post, "Morphology Post-Proc", "CC Filter (min_size=150)\n1 × 512 × 512")
    ax.annotate("", xy=(12.1, 8.85), xytext=(11.6, 8.85),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # Final Output
    draw_box(15.1, 8.4, 2.2, 0.9, c_input, "Final Output Mask", "Binary Tumor Mask\n1 × 512 × 512")
    ax.annotate("", xy=(15.1, 8.85), xytext=(14.6, 8.85),
                arrowprops=dict(arrowstyle="->", color=c_border, lw=1.5))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DEEP SUPERVISION AUXILIARY HEADS (Right of Decoders)
    # ─────────────────────────────────────────────────────────────────────────
    aux_heads = [
        ("Aux Head 0", "1×1 Conv → 1×16×16 → Interp 512", 0.7),
        ("Aux Head 1", "1×1 Conv → 1×32×32 → Interp 512", 2.4),
        ("Aux Head 2", "1×1 Conv → 1×64×64 → Interp 512", 3.9),
        ("Aux Head 3", "1×1 Conv → 1×128×128 → Interp 512", 5.4),
    ]

    for idx, (title, sub, y_pos) in enumerate(aux_heads):
        draw_box(9.6, y_pos, 2.4, 0.8, c_aux, title, sub, edge='#B45309')
        ax.annotate("", xy=(9.6, y_pos + 0.4), xytext=(9.0, y_pos + 0.4),
                    arrowprops=dict(arrowstyle="->", color='#B45309', lw=1.3, linestyle=':'))
        
        # Loss tag
        ax.text(12.3, y_pos + 0.4, f"Loss {idx+1} (w={round(0.5 - idx*0.1, 2)})", 
                fontsize=8, fontweight='bold', color='#B45309', va='center')
        ax.annotate("", xy=(12.2, y_pos + 0.4), xytext=(12.0, y_pos + 0.4),
                    arrowprops=dict(arrowstyle="->", color='#B45309', lw=1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 3. DETAILED INSET BOXES (Bottom Right: Inside DecoderBlock & sCSE)
    # ─────────────────────────────────────────────────────────────────────────
    # Inset 1: Inside DecoderBlock
    dec_panel = patches.FancyBboxPatch(
        (13.2, 3.8), 4.3, 3.7,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.5, zorder=1
    )
    ax.add_patch(dec_panel)
    ax.text(15.35, 7.2, "Inside DecoderBlock(x, skip)", ha='center', fontsize=9.5, fontweight='bold', color=c_text)

    dec_steps = [
        ("1. x_up = ConvTranspose2d(2×2, stride=2)", 6.75),
        ("2. skip_att = sCSE(skip)", 6.35),
        ("3. cat = Concatenate([skip_att, x_up])", 5.95),
        ("4. c1 = Conv2d(3×3) + BN + ReLU", 5.55),
        ("5. c2 = Conv2d(3×3) + BN + ReLU", 5.15),
        ("6. out = sCSE(c2)", 4.75),
        ("7. out = Dropout2d(p=0.10)(out)", 4.35),
    ]
    for text, y_s in dec_steps:
        ax.text(13.4, y_s, text, fontsize=8, color='#1E293B', fontfamily='monospace')

    # Inset 2: Inside Concurrent sCSE Module
    scse_panel = patches.FancyBboxPatch(
        (13.2, 0.4), 4.3, 3.2,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.5, zorder=1
    )
    ax.add_patch(scse_panel)
    ax.text(15.35, 3.3, "Inside Concurrent sCSE Module", ha='center', fontsize=9.5, fontweight='bold', color='#9A3412')

    scse_steps = [
        ("• Channel SE (cSE):", 2.9, True),
        ("  GAP → Linear(C/16) → ReLU → Linear(C) → σ", 2.6, False),
        ("• Spatial SE (sSE):", 2.25, True),
        ("  Conv2d(1×1, C→1) → Sigmoid(σ)", 1.95, False),
        ("• Concurrent Fusion:", 1.6, True),
        ("  Y_scse = cSE(X) ⊙ X  +  sSE(X) ⊙ X", 1.25, False),
    ]
    for text, y_s, is_bold in scse_steps:
        weight = 'bold' if is_bold else 'normal'
        color = '#78350F' if is_bold else '#451A03'
        ax.text(13.4, y_s, text, fontsize=7.8, fontweight=weight, color=color, fontfamily='monospace')

    # Legend / Key Attributes
    legend_text = (
        "Implementation Summary (repsegnet_v2.py):\n"
        "• Encoder: Pretrained EfficientNet-B3 (channels: 24, 32, 48, 136, 384)\n"
        "• Decoder: 5 Levels (channels: 256, 128, 64, 32, 16)\n"
        "• Skip Attention: Concurrent sCSE (Spatial + Channel Squeeze-Excitation)\n"
        "• Training Supervision: Deep Supervision across Decoder Levels 0-3\n"
        "• Total Parameters: 15.07M | GFLOPs: 8.86 | Inference: 16.45 ms (60.8 FPS)"
    )
    ax.text(0.5, 0.4, legend_text, fontsize=8.5, color='#334155',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1.2))

    plt.tight_layout()
    out_path = "outputs/repsegnet_v2_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication architecture diagram -> {out_path}")

if __name__ == '__main__':
    create_architecture_diagram()

