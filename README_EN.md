# AI-Music-Maker

**by Lombard Web Services**  
*Innovative solutions for a smarter future.*

![AI-Music-Maker](image_en.png)

AI-Music-Maker is an AI-powered music generation engine. It uses an LSTM (Long Short-Term Memory) model to compose epic orchestral pieces from your own audio files (MP3) or MIDI files. The system analyzes the notes, learns the musical style, and then generates a multi-track score with an advanced dramatic structure (Intro, Build-up, Climax, Outro) and adaptive orchestration.

---

## Features

- **Audio-to-MIDI Transcription**: Uses Spotify's Basic Pitch to extract notes from your MP3 files.
- **Multi-Track Generation**: Creates scores for multiple instruments simultaneously.
- **Adaptive Orchestration**: Automatically detects the instruments present in your source material (violin, piano, harpsichord, flute, organ, etc.) and adapts accordingly.
- **Epic Structure**: Generates a complete dramatic arc with 7 distinct sections (Intro, Awakening, Build-up 1, Build-up 2, Climax, Peak / Release, Outro), featuring progressive tempo and dynamics changes.
- **High-Quality Audio Rendering**: Converts the generated MIDI into a WAV file using FluidSynth and SoundFonts.

---

## Prerequisites and Installation

### 1. Python Environment

It is strongly recommended to use a virtual environment to isolate the project's dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it (Linux / Mac)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

### 2. Python Dependencies

Once the venv is activated (you will see `(venv)` at the beginning of your command line), install the required packages:

```bash
pip install --upgrade pip
pip install basic-pitch music21 tensorflow numpy matplotlib onnxruntime
```

**Note for Python 3.11+ users**: If `basic-pitch` has trouble installing, use the following command to force the installation of TensorFlow dependencies:

```bash
pip install basic-pitch[tf] music21 numpy matplotlib
```

### 3. Installation Verification

Test that the main modules are correctly installed:

```bash
python3 -c "import numpy; print('numpy OK')"
python3 -c "import basic_pitch; print('basic_pitch OK')"
python3 -c "import music21; print('music21 OK')"
```

### 4. FluidSynth (for audio rendering)

To convert the generated MIDI files to WAV, you need FluidSynth and a SoundFont (sound bank).

**On Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install fluidsynth fluid-soundfont-gm
```

**On macOS:**

```bash
brew install fluid-synth
```

**On Windows:**  
Download the installer from [FluidSynth.org](https://www.fluidsynth.org/) and add it to your PATH.

---

## Usage

### Step 1: Place the source files

Place your `.mp3` audio files in the `solo/` folder. The script will convert them to MIDI.

*(If you already have MIDI files, place them directly in the `midi_output/` folder.)*

### Step 2: Run the generation

Execute the main script:

```bash
python3 training.py
```

The script will successively:

1. Convert MP3 files to MIDI (if necessary).
2. Analyze the content to detect instruments.
3. Train (or load) the LSTM model on the musical sequences.
4. Generate a new composition.
5. Create a multi-track MIDI file (`output/generated_epic_orchestra.mid`).
6. Attempt an automatic audio render to `output/generated_epic_orchestra.wav`.

### Step 3: Manual rendering with FluidSynth (Optional)

If the automatic rendering fails, you can do it manually with FluidSynth:

```bash
fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 output/generated_epic_orchestra.mid -F output/generated_epic_orchestra.wav -r 44100 -g 1.0
```

---

## Troubleshooting

### Problem: `basic_pitch` fails to install or throws an error

- **Solution 1**: Make sure your `pip` is up to date.
- **Solution 2**: If the error persists, try:
  ```bash
  pip install basic-pitch[tf] music21 numpy matplotlib
  ```
- **Solution 3**: Reinstall all dependencies including `onnxruntime`:
  ```bash
  pip install --upgrade pip && pip install basic-pitch music21 tensorflow numpy matplotlib onnxruntime
  ```

### Problem: No sound is generated

- Check that `fluidsynth` is properly installed (`fluidsynth --version`).
- Verify that the path to the SoundFont is correct in the script (or in the manual command).

### Problem: All instruments sound like a violin

This issue occurred with older versions of music21. Make sure you are using the provided version of the `gen-solo-instruments.py` script, which forces MIDI channel reassignment via `mido`.

---

## License & Copyright

Copyright © 2026 Lombard Web Services. All rights reserved.
