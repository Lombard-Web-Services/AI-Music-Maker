#!/usr/bin/env python
# coding: utf-8

"""
Génère 3 variantes WAV de output/generated_epic_orchestra.mid :
  1. piano + violin
  2. harpsichord + contrabass + flute
  3. organ

Méthode robuste :
  - music21 réassigne les instruments (structure)
  - mido force les program_change sur chaque canal (fiabilité)
  - vérification post-export des programmes réellement écrits
"""

import os
import sys
import signal
import atexit
import subprocess
from music21 import converter, instrument

import mido  # pip install mido

# ------------------------------------------------------------
# Ctrl+C propre
# ------------------------------------------------------------
_CHILD_PROCESSES = []
_SHUTTING_DOWN  = False

def _cleanup_children():
    for p in list(_CHILD_PROCESSES):
        try:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=2)
                except subprocess.TimeoutExpired: p.kill()
        except Exception: pass
    _CHILD_PROCESSES.clear()

def _hard_exit(code=130):
    _cleanup_children()
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception: pass
    os._exit(code)

def _signal_handler(signum, frame):
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN: _hard_exit(130)
    _SHUTTING_DOWN = True
    print(f"\n[INFO] Signal {signum} reçu → arrêt propre…")
    _hard_exit(130)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_children)

def run_cmd(cmd, timeout=300):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    _CHILD_PROCESSES.append(p)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except KeyboardInterrupt:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
        try: p.wait(timeout=3)
        except Exception:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
        raise
    finally:
        if p in _CHILD_PROCESSES: _CHILD_PROCESSES.remove(p)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
INPUT_MIDI  = "output/generated_epic_orchestra.mid"
OUTPUT_DIR  = "output"
SOUNDFONTS  = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    os.path.expanduser("~/soundfonts/GeneralUser_GS.sf2"),
    os.path.expanduser("~/GeneralUser_GS.sf2"),
    "/usr/share/sounds/sf2/GeneralUser_GS.sf2",
]

ROLE_ORDER = ["bass", "counter", "melody", "harmony", "brass", "percussion"]

# Variantes : rôle → programme GM
# (le nom d'instrument est indicatif, on passe par le numéro GM)
ROLE_TO_VARIANT = {
    "v1_piano_violin": {
        "bass":    0,   # Piano
        "counter": 40,  # Violin
        "melody":  0,   # Piano
        "harmony": 0,   # Piano
        "brass":   40,  # Violin
        "percussion": 0,
    },
    "v2_harpsi_bass_flute": {
        "bass":    43,  # Contrabass
        "counter": 6,   # Harpsichord
        "melody":  73,  # Flute
        "harmony": 6,   # Harpsichord
        "brass":   73,  # Flute
        "percussion": 0,
    },
    "v3_organ": {
        "bass":    19,  # Church Organ
        "counter": 16,  # Drawbar Organ
        "melody":  19,  # Church Organ
        "harmony": 16,  # Drawbar Organ
        "brass":   19,  # Church Organ
        "percussion": 0,
    },
}

INSTRUMENT_TO_ROLE = {
    "Violin":      "melody",
    "Flute":       "melody",
    "Trumpet":     "brass",
    "Viola":       "counter",
    "Saxophone":   "counter",
    "Horn":        "counter",
    "Violoncello": "bass",
    "Contrabass":  "bass",
    "Harpsichord": "harmony",
    "Guitar":      "harmony",
    "Piano":       "harmony",
    "Organ":       "harmony",
}

def pick_soundfont():
    for p in SOUNDFONTS:
        if os.path.isfile(p): return p
    return None

# ------------------------------------------------------------
# Détection du rôle
# ------------------------------------------------------------
def guess_role(part, index):
    try:
        pid = str(part.id).lower() if part.id is not None else ""
    except Exception:
        pid = ""
    for role in ROLE_ORDER:
        if role in pid:
            return role

    for inst in part.recurse().getElementsByClass(instrument.Instrument):
        iname = type(inst).__name__
        if iname in INSTRUMENT_TO_ROLE:
            return INSTRUMENT_TO_ROLE[iname]

    if index < len(ROLE_ORDER):
        return ROLE_ORDER[index]
    return ROLE_ORDER[0]

# ------------------------------------------------------------
# Réassignation via mido (fiable)
# ------------------------------------------------------------
def reassign_programs_mido(input_path, output_path, role_programs):
    """
    Ouvre le MIDI avec mido, détecte le rôle de chaque piste via son nom,
    puis force le program_change sur le canal correspondant.
    """
    midi = mido.MidiFile(input_path)

    # 1) Détecter le rôle de chaque piste
    track_roles = {}
    for i, track in enumerate(midi.tracks):
        name = ""
        for msg in track:
            if msg.type == "track_name":
                name = msg.name.lower()
                break
        role = None
        for r in ROLE_ORDER:
            if r in name:
                role = r
                break
        if role is None and i < len(ROLE_ORDER):
            role = ROLE_ORDER[i]
        track_roles[i] = role or ROLE_ORDER[0]
        print(f"    Piste {i:2d} : name='{name or '(vide)'}' → rôle={track_roles[i]}")

    # 2) Trouver le canal utilisé par chaque piste
    track_channels = {}
    for i, track in enumerate(midi.tracks):
        ch = None
        for msg in track:
            if hasattr(msg, "channel") and msg.type in ("note_on", "note_off"):
                ch = msg.channel
                break
        if ch is None:
            for msg in track:
                if hasattr(msg, "channel"):
                    ch = msg.channel
                    break
        track_channels[i] = ch if ch is not None else i
        print(f"    Piste {i:2d} : canal={track_channels[i]}")

    # 3) S'assurer que chaque piste utilise un canal unique
    used = set()
    for i in midi.tracks:
        pass
    for i in range(len(midi.tracks)):
        ch = track_channels[i]
        if ch in used:
            # Trouve un canal libre
            for candidate in range(16):
                if candidate != 9 and candidate not in used:
                    ch = candidate
                    break
        used.add(ch)
        track_channels[i] = ch

    # 4) Réécrire : ajouter/remplacer program_change au début de chaque piste
    for i, track in enumerate(midi.tracks):
        role = track_roles[i]
        program = role_programs.get(role, 0)
        ch = track_channels[i]

        # Remplace les program_change existants
        new_track = mido.MidiTrack()
        inserted = False
        for msg in track:
            # Réécrit le canal des notes sur le bon canal
            if hasattr(msg, "channel") and msg.type in (
                "note_on", "note_off", "control_change", "program_change",
                "pitchwheel", "aftertouch", "polytouch"
            ):
                msg = msg.copy(channel=ch)

            # Insère le nouveau program_change au début
            if not inserted and msg.type not in ("track_name", "text", "copyright", "time_signature", "key_signature", "set_tempo"):
                new_track.append(mido.Message("program_change", program=program, channel=ch))
                inserted = True

            if msg.type == "program_change":
                continue  # supprime l'ancien
            new_track.append(msg)

        if not inserted:
            new_track.insert(0, mido.Message("program_change", program=program, channel=ch))

        midi.tracks[i] = new_track
        print(f"    Piste {i:2d} → program_change {program:3d} sur canal {ch}")

    midi.save(output_path)
    print(f"  [OK] MIDI réassigné : {output_path}")

# ------------------------------------------------------------
# Vérification post-export
# ------------------------------------------------------------
def verify_programs(midi_path):
    midi = mido.MidiFile(midi_path)
    programs = {}
    for i, track in enumerate(midi.tracks):
        for msg in track:
            if msg.type == "program_change":
                programs[i] = (msg.channel, msg.program)
                break
    print(f"  [VERIF] Programmes réellement écrits dans {os.path.basename(midi_path)} :")
    for track_idx, (ch, prog) in programs.items():
        print(f"    Piste {track_idx} : canal {ch} → programme {prog}")
    return programs

# ------------------------------------------------------------
# Rendu WAV
# ------------------------------------------------------------
def render_wav(midi_path, wav_path, soundfont):
    cmd = ["fluidsynth", "-ni", soundfont, midi_path,
           "-F", wav_path, "-r", "44100", "-g", "1.0"]
    print(f"  → fluidsynth {os.path.basename(wav_path)}")
    rc, out, err = run_cmd(cmd, timeout=300)
    if rc == 0 and os.path.exists(wav_path):
        print(f"    [OK] {wav_path}")
        return True
    print(f"    [ERREUR] {err[:300]}")
    return False

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not os.path.isfile(INPUT_MIDI):
        raise FileNotFoundError(f"MIDI introuvable : {INPUT_MIDI}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    soundfont = pick_soundfont()
    if not soundfont:
        print("[ERREUR] Aucun soundfont trouvé.")
        print("  sudo apt install fluidsynth fluid-soundfont-gm")
        return
    print(f"Soundfont : {soundfont}\n")

    try:
        subprocess.run(["fluidsynth", "--version"],
                       capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[ERREUR] fluidsynth non installé.")
        print("  sudo apt install fluidsynth")
        return

    for variant_name, role_programs in ROLE_TO_VARIANT.items():
        print("=" * 70)
        print(f"Variante : {variant_name}")
        print("=" * 70)

        variant_midi = os.path.join(OUTPUT_DIR, f"{variant_name}.mid")
        variant_wav  = os.path.join(OUTPUT_DIR, f"{variant_name}.wav")

        reassign_programs_mido(INPUT_MIDI, variant_midi, role_programs)
        verify_programs(variant_midi)
        render_wav(variant_midi, variant_wav, soundfont)
        print()

    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    for name in ROLE_TO_VARIANT:
        print(f"  {OUTPUT_DIR}/{name}.wav")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrompu.")
        _hard_exit(130)
