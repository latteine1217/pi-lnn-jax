from __future__ import annotations

import base64
import os
import sys
from pathlib import Path


OUTPUT_PATH = Path(
    "/Users/latteine/Documents/coding/pi-lnn-jax/paper/tmlr-format/figures/architecture_imagegen.png"
)

PROMPT = """Create a paper-quality landscape PNG architecture diagram for the PI-CON neural operator.

Use case: scientific-educational
Asset type: academic paper architecture figure
Style: horizontal left-to-right dataflow diagram, clean academic paper style, white background, rounded-rectangle labeled boxes, directional arrows with labels, minimal and uncluttered, sans-serif labels, crisp arrows.
Canvas: landscape 1536x1024.

Visual rules:
- Inherited DeepONet backbone boxes are grey.
- Three NEW PI-CON components are highlighted in light blue with numbered circular badges 1, 2, and 3.
- Keep all labels readable and concise.
- Use zone titles above grouped regions.
- Branch and trunk merge only at Cross-Attention / Fusion.
- Never connect Branch Basis or Trunk Basis directly to Field.

Topology:

Top row zone title: "Branch path: sparse sensor memory"
Observations
Label inside box: "Observations\\nK=100 sensor values [T,K,{u,v}]\\npositions + time"
Arrow label: "encode"
Spatial Tokens
Label inside box: "Spatial Tokens\\nFourier/RFF encoding\\nresidual MLP"
Arrow label: "scan"
Liquid CfC Memory
Label inside box: "Liquid CfC Memory\\ntoken attention\\ncausal scan\\ninput-dependent tau"
This Liquid CfC Memory box is NEW, light blue, badge 1.

Lower-left zone title: "Trunk path: query feature"
Query
Label inside box: "Query\\n(x, y, t_q, c)"
Arrow label: "embed"
Trunk Feature
Label inside box: "Trunk Feature\\nFourier + temporal anchor\\ndt_to_query"

Center zone title: "Interaction"
Cross-Attention Readout
Label inside box: "Cross-Attention Readout\\nQ from Trunk Feature\\nK,V from Liquid CfC Memory\\nisotropic distance bias"
This Cross-Attention Readout box is NEW, light blue, badge 2.
Connect Liquid CfC Memory to Cross-Attention Readout.
Connect Trunk Feature to Cross-Attention Readout.

Right zone title: "Output and loss"
From Cross-Attention Readout, split into two grey boxes:
Branch Basis
Trunk Basis
Both arrows then merge into:
DeepONet Fusion
Label inside box: "DeepONet Fusion\\nbranch x trunk basis"
Arrow to:
Field
Label inside box: "Field\\n(u, v, p)\\np physics-only"
Arrow label: "autograd"
Loss
Label inside box: "Loss\\nAL hard constraint div(u)=0\\nsensor MSE + NS residual\\nGradNorm balancing"
This Loss box is NEW, light blue, badge 3.

Bottom row optimizer feedback:
Use grey boxes and dashed arrows:
Loss --optimize--> SOAP + Schedule-Free
Label inside box: "SOAP + Schedule-Free\\npreconditioned updates"
SOAP + Schedule-Free --update theta--> model parameters theta
model parameters theta --> trainable modules
Connect the dashed feedback visually back toward the trainable modules without clutter.

Bottom annotations:
Formula: "L = L_data(u,v) + GradNorm(NS_u, NS_v, cont) + AL(cont)"
Note: "Full-field DNS is not used as supervision."

Legend:
"grey = inherited DeepONet backbone"
"blue with badge = added by PI-CON (1,2,3)"

Avoid photorealism, 3D rendering, decorative backgrounds, tiny unreadable labels, extra modules, and any direct Branch Basis or Trunk Basis arrow into Field.
"""


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing", file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI()
    response = client.images.generate(
        model="gpt-image-1",
        size="1536x1024",
        quality="high",
        prompt=PROMPT,
    )

    image_b64 = response.data[0].b64_json
    if not image_b64:
        print("Image API response did not include data[0].b64_json", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(base64.b64decode(image_b64))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Bytes: {OUTPUT_PATH.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
