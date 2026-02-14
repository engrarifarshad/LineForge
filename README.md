# LineForge
Reconstructing images using thousands of intelligent straight lines — where structure, contrast, and geometry transform pixels into generative art.


✨ Overview

LineForge is a generative art project that rebuilds grayscale images using nothing but straight lines.

Instead of displaying pixels directly, this system analyzes image structure, contrast, and gradients — then reconstructs the image using thousands of strategically placed lines.

The result is a portrait that looks photographic from a distance…
but reveals pure geometry up close.

🧠 Concept

The core idea is simple:

Read an input grayscale image

Analyze visual importance (contrast + edges + gradients)

Use feature-aware sampling

Rebuild the image using only straight lines

No curves.
No brushes.
No shading tools.

Just geometry guided by image intelligence.

🎯 Key Features

Feature-aware line placement

Edge-following directional strokes

Multi-orientation hatching for tonal depth

Adaptive line length based on local contrast

High-contrast region reinforcement

Adjustable number of lines for different detail levels

🖼 How It Works (High-Level)

The system combines three rendering strategies:

1️⃣ Tone-Based Hatching

Darker regions receive more lines.
Multiple orientations (0°, 45°, 90°, 135°) create depth and texture.

2️⃣ Edge-Following Lines

Lines align along structural boundaries to preserve shape and form.

3️⃣ Feature Connections

High-contrast points are selectively connected to retain important details.

Together, these techniques allow the reconstruction to preserve both:

Low-level details (shading & tone)

High-level structure (edges & facial features)

🚀 Usage
1. Clone the repository
git clone https://github.com/your-username/LineForge.git
cd LineForge

2. Install dependencies
pip install opencv-python numpy

3. Run the script
python main.py


Make sure your input image path is correctly set inside the script.

⚙️ Parameters You Can Tune

You can experiment with:

num_lines → Controls detail density

line_thickness → Adjust stroke weight

Edge thresholds → Control structural emphasis

Higher line counts = More detail
Lower line counts = More abstract look

📸 Example

Input: Grayscale portrait
Output: 50,000+ intelligently placed straight lines forming the same portrait

(You can add before/after images here in the repo)

🎨 Why This Project?

This project explores the intersection of:

Computer Vision

Generative Art

Feature-Based Rendering

Algorithmic Creativity

It demonstrates how mathematical structure can transform simple pixel data into expressive geometric art.

💡 Future Improvements

Color-based rendering

Directional coherence refinement

GPU acceleration

Real-time rendering

Interactive parameter control

🤝 Contributing

Ideas, optimizations, and creative experiments are welcome.
Feel free to fork the project and push it further.

📜 License

MIT License (or choose your preferred license)
