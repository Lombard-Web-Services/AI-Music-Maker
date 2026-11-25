#!/usr/bin/env python
# coding: utf-8

"""
Génération d'une musique ÉPIQUE MULTIPISTE avec structure fine — VERSION AUTO-ADAPTATIVE.

Pipeline : MP3 → MIDI (Basic Pitch) → notes (music21) → LSTM → MIDI orchestral multipiste

Détection d'instruments :
  - Programme GM déclaré (peu fiable avec Basic Pitch : toujours 0–4)
  - + Analyse REGISTRE + POLYPHONIE systématique quand générique
  - Percussion uniquement si détectée (canal 10 ou GM 112–119)

Gestion Ctrl+C :
  - Handler SIGINT/SIGTERM qui tue les sous-process (fluidsynth, basic-pitch)
  - os._exit(130) pour tuer les threads natifs TF/XLA
  - Aucun process résiduel après arrêt

Mode d'entrée :
  - Si solo/*.mp3 existe → conversion Basic Pitch puis pipeline complet
  - Sinon → utilise directement les MIDI déjà présents dans midi_output/
"""

import os
import sys
import glob
import signal
import atexit
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt

from music21 import (
    converter, instrument, note, chord, stream,
    tempo, meter, dynamics
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ============================================================
# GESTION PROPRE DE CTRL+C + SOUS-PROCESS
# ============================================================
_CHILD_PROCESSES = []
_SHUTTING_DOWN  = False


def _cleanup_children():
    for p in list(_CHILD_PROCESSES):
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=1)
        except Exception:
            pass
    _CHILD_PROCESSES.clear()


def _hard_exit(code=130):
    _cleanup_children()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def _signal_handler(signum, frame):
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        _hard_exit(130)
    _SHUTTING_DOWN = True
    print(f"\n[INFO] Signal {signum} reçu → arrêt propre en cours…")
    print("[INFO] (Appuyez à nouveau sur Ctrl+C pour forcer)")
    _hard_exit(130)


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_cleanup_children)


def run_cmd(cmd, timeout=600):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    _CHILD_PROCESSES.append(p)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except KeyboardInterrupt:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        raise
    finally:
        if p in _CHILD_PROCESSES:
            _CHILD_PROCESSES.remove(p)


# ============================================================
# CONFIGURATION
# ============================================================
SOLO_DIR        = "solo"
MIDI_DIR        = "midi_output"
MODEL_DIR       = "model_checkpoints"
OUTPUT_DIR      = "output"

SEQUENCE_LENGTH = 32
EPOCHS          = 80
BATCH_SIZE      = 64
LSTM_UNITS      = 256
DROPOUT_RATE    = 0.3
GENERATE_NOTES  = 360
TEMPERATURE     = 0.72

BASE_TEMPO = 86

# ============================================================
# STRUCTURE FINE — rôles abstraits, résolus dynamiquement
# ============================================================
SECTIONS = [
    {"name": "Intro",          "ratio": 0.12, "roles": ["melody", "bass", "harmony"],
     "density": 0.35, "tempo": 78, "dynamic": "pp", "percussion": False},
    {"name": "Awakening",      "ratio": 0.13, "roles": ["melody", "counter", "bass", "harmony"],
     "density": 0.50, "tempo": 82, "dynamic": "p", "crescendo_to": "mp", "percussion": False},
    {"name": "Build-up 1",     "ratio": 0.15, "roles": ["melody", "counter", "bass", "harmony"],
     "density": 0.68, "tempo": 88, "dynamic": "mp", "crescendo_to": "mf",
     "percussion": True, "percussion_intensity": 0.35},
    {"name": "Build-up 2",     "ratio": 0.15, "roles": ["melody", "counter", "bass", "brass"],
     "density": 0.82, "tempo": 94, "dynamic": "mf", "crescendo_to": "f",
     "percussion": True, "percussion_intensity": 0.65},
    {"name": "Climax",         "ratio": 0.20, "roles": ["melody", "counter", "bass", "brass", "harmony"],
     "density": 1.00, "tempo": 98, "dynamic": "ff",
     "percussion": True, "percussion_intensity": 1.00},
    {"name": "Peak / Release", "ratio": 0.13, "roles": ["melody", "counter", "bass", "brass"],
     "density": 0.75, "tempo": 92, "dynamic": "f", "crescendo_to": "mf",
     "percussion": True, "percussion_intensity": 0.55},
    {"name": "Outro",          "ratio": 0.12, "roles": ["melody", "bass", "harmony"],
     "density": 0.40, "tempo": 76, "dynamic": "p", "crescendo_to": "pp", "percussion": False},
]

# Plage dynamique adoucie (option A) : évite l'effet « son qui s'éteint »
VELOCITY_MAP = {
    "ppp": 45, "pp": 55, "p": 65, "mp": 72,
    "mf": 80, "f": 90, "ff": 105, "fff": 115,
}

GENERIC_PROGRAMS = {0, 1, 2, 3, 4}

GM_PROGRAM_TO_INSTRUMENT = {
    6:  instrument.Harpsichord, 7:  instrument.Harpsichord, 8: instrument.Harpsichord,
    24: instrument.Guitar,      25: instrument.Guitar,     26: instrument.Guitar,
    32: instrument.Contrabass,  33: instrument.Contrabass, 34: instrument.Contrabass,
    35: instrument.Contrabass,  36: instrument.Contrabass, 37: instrument.Contrabass,
    38: instrument.Contrabass,  39: instrument.Contrabass,
    40: instrument.Violin,      41: instrument.Viola,      42: instrument.Violoncello,
    43: instrument.Contrabass,  44: instrument.Violin,     45: instrument.Viola,
    46: instrument.Violoncello, 47: instrument.Contrabass,
    56: instrument.Trumpet,     57: instrument.Trumpet,    58: instrument.Trumpet,
    59: instrument.Trumpet,     60: instrument.Horn,       61: instrument.Horn,
    62: instrument.Horn,        63: instrument.Horn,
    64: instrument.Saxophone,   65: instrument.Saxophone,  66: instrument.Saxophone,
    67: instrument.Saxophone,   68: instrument.Saxophone,  69: instrument.Saxophone,
    70: instrument.Saxophone,   71: instrument.Saxophone,
    72: instrument.Flute,       73: instrument.Flute,      74: instrument.Flute,
    75: instrument.Flute,       76: instrument.Flute,      77: instrument.Flute,
    78: instrument.Flute,       79: instrument.Flute,
    80: instrument.Trumpet,     81: instrument.Trumpet,    82: instrument.Trumpet,
    83: instrument.Trumpet,     84: instrument.Trumpet,    85: instrument.Trumpet,
    86: instrument.Trumpet,     87: instrument.Trumpet,
    104: instrument.Guitar,     105: instrument.Guitar,    106: instrument.Guitar,
    107: instrument.Guitar,     108: instrument.Guitar,    109: instrument.Guitar,
    110: instrument.Guitar,     111: instrument.Guitar,
}

PERCUSSION_PROGRAMS = set(range(112, 120))
PERCUSSION_CHANNEL  = 9

SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    os.path.expanduser("~/soundfonts/GeneralUser_GS.sf2"),
    os.path.expanduser("~/GeneralUser_GS.sf2"),
    "/usr/share/sounds/sf2/GeneralUser_GS.sf2",
    "GeneralUser_GS.sf2",
]

os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
ORCH_PATH  = os.path.join(MODEL_DIR, "orchestration.pkl")

# ============================================================
# ÉTAPE 1 : CONVERSION MP3 → MIDI (OPTIONNELLE)
# ============================================================
print("=" * 70)
print("ÉTAPE 1 : Conversion MP3 → MIDI (Basic Pitch) — optionnelle")
print("=" * 70)

mp3_files = sorted(glob.glob(os.path.join(SOLO_DIR, "*.mp3")))

if not mp3_files:
    print(f"[INFO] Aucun MP3 dans {SOLO_DIR}/ → on utilise les MIDI existants.")
else:
    print(f"{len(mp3_files)} fichiers MP3 trouvés.")
    for mp3_path in mp3_files:
        basename = os.path.splitext(os.path.basename(mp3_path))[0]
        midi_path = os.path.join(MIDI_DIR, basename + "_basic_pitch.mid")

        if os.path.exists(midi_path):
            print(f"  [SKIP] {basename}")
            continue

        print(f"  Conversion de {basename}...")
        try:
            rc, out, err = run_cmd(
                ["basic-pitch", MIDI_DIR, mp3_path, "--model-serialization", "onnx"],
                timeout=600,
            )
            if rc != 0:
                print(f"    [ERREUR] {err[:250]}")
            else:
                print(f"    [OK]")
        except FileNotFoundError:
            raise RuntimeError("Installez basic-pitch : pip install basic-pitch")
        except KeyboardInterrupt:
            print("\n[INFO] Conversion interrompue.")
            raise

print("Conversion terminée.\n")

# ------------------------------------------------------------
# VÉRIFICATION : au moins un MIDI disponible ?
# ------------------------------------------------------------
midi_files = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
if not midi_files:
    raise ValueError(
        f"Aucun MIDI dans {MIDI_DIR}/.\n"
        f"  → Soit ajoute des MP3 dans {SOLO_DIR}/ (conversion automatique),\n"
        f"  → Soit copie des fichiers .mid dans {MIDI_DIR}/."
    )
print(f"{len(midi_files)} fichiers MIDI disponibles.\n")

# ============================================================
# ÉTAPE 1bis : DÉTECTION AUTOMATIQUE DE L'ORCHESTRATION
# ============================================================
print("=" * 70)
print("ÉTAPE 1bis : Détection automatique des instruments présents")
print("=" * 70)


def _pitches_of(el):
    try:
        if isinstance(el, note.Note):
            return [el.pitch.midi]
        if isinstance(el, chord.Chord):
            return [p.midi for p in el.pitches]
    except Exception:
        pass
    return []


def _collect_pitches_and_programs(midi_files):
    programs = set()
    all_pitches = []
    polyphonic_hits = 0
    total_events = 0
    has_percussion = False

    for midi_path in midi_files:
        try:
            midi = converter.parse(midi_path)
            parts = instrument.partitionByInstrument(midi)
            iter_parts = parts.parts if parts else [midi]

            for p in iter_parts:
                for inst in p.recurse().getElementsByClass(instrument.Instrument):
                    prog = getattr(inst, "midiProgram", None)
                    if prog is not None:
                        programs.add(int(prog))
                        if int(prog) in PERCUSSION_PROGRAMS:
                            has_percussion = True

                for el in p.recurse().notes:
                    pitches = _pitches_of(el)
                    if pitches:
                        all_pitches.extend(pitches)
                        total_events += 1
                        if isinstance(el, chord.Chord) and len(pitches) >= 2:
                            polyphonic_hits += 1
        except Exception as e:
            print(f"  [WARN] {os.path.basename(midi_path)} : {e}")

    try:
        import mido
        for midi_path in midi_files:
            m = mido.MidiFile(midi_path)
            for track in m.tracks:
                for msg in track:
                    if getattr(msg, "channel", None) == PERCUSSION_CHANNEL \
                       and msg.type in ("note_on", "note_off") \
                       and getattr(msg, "velocity", 0) > 0:
                        has_percussion = True
                        break
                if has_percussion:
                    break
            if has_percussion:
                break
    except Exception:
        pass

    polyphony_rate = (polyphonic_hits / total_events) if total_events else 0.0
    return {
        "programs": programs,
        "pitches": all_pitches,
        "polyphony_rate": polyphony_rate,
        "has_percussion": has_percussion,
    }


def _register_roles_from_pitches(pitches):
    roles = {}
    if not pitches:
        return roles
    arr = np.array(pitches)
    p10, p30, p50, p70, p90 = np.percentile(arr, [10, 30, 50, 70, 90]).astype(int)
    print(f"  Registres (p10/p30/p50/p70/p90) : {p10}/{p30}/{p50}/{p70}/{p90}")

    roles["bass"] = instrument.Contrabass() if p10 < 36 else instrument.Violoncello()
    roles["counter"] = instrument.Viola()
    roles["melody"] = instrument.Flute() if p90 > 90 else instrument.Violin()
    return roles


def detect_instruments(midi_files):
    info = _collect_pitches_and_programs(midi_files)
    programs = info["programs"]
    pitches = info["pitches"]
    polyphony_rate = info["polyphony_rate"]
    has_percussion = info["has_percussion"]

    print(f"  Programmes GM détectés : {sorted(programs) if programs else 'aucun'}")
    print(f"  Taux de polyphonie     : {polyphony_rate:.2%}")
    print(f"  Percussion détectée    : {has_percussion}")

    detected = {}
    seen_classes = set()
    explicit_only_generic = programs.issubset(GENERIC_PROGRAMS) if programs else True

    for prog in sorted(programs):
        if prog in GENERIC_PROGRAMS or prog in PERCUSSION_PROGRAMS:
            continue
        cls = GM_PROGRAM_TO_INSTRUMENT.get(prog)
        if cls is None or cls in seen_classes:
            continue
        seen_classes.add(cls)
        inst = cls()
        if isinstance(inst, (instrument.Violin, instrument.Flute)):
            detected.setdefault("melody", inst)
        elif isinstance(inst, instrument.Trumpet):
            detected.setdefault("melody", inst)
            detected.setdefault("brass", inst)
        elif isinstance(inst, (instrument.Viola, instrument.Saxophone,
                               instrument.Violoncello, instrument.Horn)):
            detected.setdefault("counter", inst)
        elif isinstance(inst, instrument.Contrabass):
            detected.setdefault("bass", inst)
        elif isinstance(inst, (instrument.Harpsichord, instrument.Guitar)):
            detected.setdefault("harmony", inst)

    if explicit_only_generic:
        for role, inst in _register_roles_from_pitches(pitches).items():
            detected.setdefault(role, inst)
        if polyphony_rate >= 0.15:
            detected.setdefault("harmony", instrument.Harpsichord())
        if pitches:
            arr = np.array(pitches)
            if arr.max() > 88 and len(pitches) > 2000:
                detected.setdefault("brass", instrument.Trumpet())
                detected.setdefault("counter", instrument.Horn())

    if not detected:
        print("  [INFO] Aucun instrument reconnu → fallback par registre.")
        for role, inst in _register_roles_from_pitches(pitches).items():
            detected.setdefault(role, inst)
        if not detected:
            detected = {
                "bass":    instrument.Violoncello(),
                "counter": instrument.Viola(),
                "melody":  instrument.Violin(),
                "harmony": instrument.Harpsichord(),
            }

    if has_percussion:
        detected["percussion"] = instrument.Timpani()

    return {
        "detected": detected,
        "has_percussion": has_percussion,
        "fallback": explicit_only_generic,
        "raw_programs": programs,
        "polyphony_rate": polyphony_rate,
    }


if os.path.exists(ORCH_PATH):
    with open(ORCH_PATH, "rb") as f:
        orch_info = pickle.load(f)
    print(f"Orchestration chargée depuis {ORCH_PATH}")
    print("  ⚠  Si tu viens de changer le code de détection, supprime ce cache !")
else:
    orch_info = detect_instruments(midi_files)
    with open(ORCH_PATH, "wb") as f:
        pickle.dump(orch_info, f)
    print(f"Orchestration sauvegardée dans {ORCH_PATH}")

DETECTED     = orch_info["detected"]
HAS_PERCO    = orch_info["has_percussion"]
IS_FALLBACK  = orch_info["fallback"]

print("\nInstrumentation retenue :")
for role, inst in DETECTED.items():
    print(f"  {role:12} → {type(inst).__name__}")
if not HAS_PERCO:
    print("  percussion   → (absente du training, pas de piste générée)")
print()

# ============================================================
# ÉTAPE 2 : EXTRACTION DES NOTES
# ============================================================
print("=" * 70)
print("ÉTAPE 2 : Extraction des notes (music21)")
print("=" * 70)


def extract_notes_from_midi(midi_path):
    notes_list = []
    try:
        midi = converter.parse(midi_path)
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes_list.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes_list.append('.'.join(str(n) for n in element.normalOrder))
    except Exception as e:
        print(f"  [ERREUR] {os.path.basename(midi_path)} : {e}")
    return notes_list


if os.path.exists(VOCAB_PATH) and os.path.exists(MODEL_PATH):
    print("Vocabulaire + modèle déjà présents → chargement")
    with open(VOCAB_PATH, "rb") as f:
        vocab, note_to_int, int_to_note = pickle.load(f)
    print(f"Vocabulaire : {len(vocab)} symboles")
    all_notes = None
else:
    all_notes = []
    print(f"{len(midi_files)} fichiers MIDI à parser")
    for midi_path in midi_files:
        notes = extract_notes_from_midi(midi_path)
        all_notes.extend(notes)
        print(f"  {os.path.basename(midi_path)} → {len(notes)} notes")

    if not all_notes:
        raise ValueError("Aucune note extraite. Vérifiez la conversion.")

    print(f"\nTotal notes : {len(all_notes)}")
    print(f"Vocabulaire unique : {len(set(all_notes))}")

    vocab = sorted(set(all_notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for i, n in enumerate(vocab)}

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump((vocab, note_to_int, int_to_note), f)

# ============================================================
# ÉTAPE 3 : SÉQUENCES
# ============================================================
print("\n" + "=" * 70)
print("ÉTAPE 3 : Création des séquences")
print("=" * 70)

if all_notes is None:
    all_notes = []
    for midi_path in midi_files:
        all_notes.extend(extract_notes_from_midi(midi_path))

network_input = []
network_output = []

for i in range(0, len(all_notes) - SEQUENCE_LENGTH):
    seq_in = all_notes[i:i + SEQUENCE_LENGTH]
    seq_out = all_notes[i + SEQUENCE_LENGTH]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
print(f"{n_patterns} séquences créées")

network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
network_input = network_input / float(len(vocab))
network_output = to_categorical(network_output, num_classes=len(vocab))

# ============================================================
# ÉTAPE 4-5 : MODÈLE
# ============================================================
if os.path.exists(MODEL_PATH):
    print("\n" + "=" * 70)
    print("ÉTAPE 4-5 : Modèle déjà entraîné → chargement")
    print("=" * 70)
    model = load_model(MODEL_PATH)
    print(f"Chargé : {MODEL_PATH}")
else:
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Construction du modèle LSTM")
    print("=" * 70)

    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(len(vocab), activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    model.summary()

    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Entraînement")
    print("=" * 70)

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "weights.keras"),
        monitor='loss', verbose=1, save_best_only=True, mode='min'
    )

    try:
        history = model.fit(
            network_input, network_output,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[checkpoint], verbose=1
        )
        model.save(MODEL_PATH)
        print("Modèle sauvegardé.")

        plt.figure(figsize=(9, 4))
        plt.plot(history.history['loss'], label='loss')
        plt.title("Courbe de loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
        plt.close()
    except KeyboardInterrupt:
        try:
            model.save(MODEL_PATH + ".partial")
            print(f"\n[INFO] Modèle partiel sauvegardé : {MODEL_PATH}.partial")
        except Exception as e:
            print(f"\n[WARN] Impossible de sauvegarder le modèle partiel : {e}")
        raise

# ============================================================
# GÉNÉRATION D'UNE SÉQUENCE
# ============================================================
def generate_sequence(model, network_input, vocab_size, length=200, temperature=0.75):
    start_idx = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_idx].copy()
    generated = []

    for _ in range(length):
        pred = model.predict(np.expand_dims(pattern, axis=0), verbose=0)[0]
        pred = np.log(np.asarray(pred).astype('float64') + 1e-8) / temperature
        pred = np.exp(pred)
        pred /= np.sum(pred)
        idx = int(np.argmax(np.random.multinomial(1, pred, 1)))
        generated.append(idx)
        new_val = np.array([[idx / float(vocab_size)]])
        pattern = np.concatenate((pattern[1:], new_val), axis=0)

    return generated

# ============================================================
# ÉTAPE 6 : GÉNÉRATION DES SÉQUENCES
# ============================================================
print("\n" + "=" * 70)
print("ÉTAPE 6 : Génération des séquences (adaptée à l'orchestration détectée)")
print("=" * 70)

role_sequences = {}
for role, inst in DETECTED.items():
    if role == "percussion":
        continue
    length = GENERATE_NOTES // 2 if role == "bass" else GENERATE_NOTES
    temp = TEMPERATURE + {"melody": 0.00, "counter": 0.07,
                          "harmony": 0.04, "brass": 0.02,
                          "bass": -0.17}.get(role, 0.05)
    print(f"  → {role:10} ({type(inst).__name__:20}) temp={temp:.2f} len={length}")
    role_sequences[role] = generate_sequence(
        model, network_input, len(vocab), length=length, temperature=temp
    )

# ============================================================
# UTILITAIRES
# ============================================================
def make_note_or_chord(note_str, instr, offset, duration=0.5, octave_shift=0, velocity=80):
    try:
        if '.' in note_str:
            pitches = [int(p) for p in note_str.split('.')]
            notes = []
            for p in pitches:
                n = note.Note(p)
                if octave_shift != 0:
                    n.octave = max(1, min(7, n.octave + octave_shift))
                n.storedInstrument = instr
                n.volume.velocity = velocity
                notes.append(n)
            c = chord.Chord(notes)
            c.offset = offset
            c.quarterLength = duration
            return c
        else:
            n = note.Note(note_str)
            if octave_shift != 0:
                n.octave = max(1, min(7, n.octave + octave_shift))
            n.storedInstrument = instr
            n.volume.velocity = velocity
            n.offset = offset
            n.quarterLength = duration
            return n
    except Exception:
        return None


def apply_density(indices, density):
    if density >= 0.99:
        return indices
    return [idx if np.random.random() < density else None for idx in indices]

# ============================================================
# ÉTAPE 7 : CONSTRUCTION DU SCORE MULTIPISTE
# ============================================================
print("\n" + "=" * 70)
print("ÉTAPE 7 : Construction du score multipiste")
print("=" * 70)

score = stream.Score()
score.insert(0, meter.TimeSignature('4/4'))

parts = {}
for role, inst in DETECTED.items():
    p = stream.Part(id=role.capitalize())
    p.insert(0, inst)
    parts[role] = p
    print(f"  Partie créée : {role:12} ({type(inst).__name__})")

if not HAS_PERCO:
    print("  (aucune piste percussion — absente du training)")

total_notes = GENERATE_NOTES
section_bounds = []
current = 0
for sec in SECTIONS:
    length = int(total_notes * sec["ratio"])
    section_bounds.append((current, current + length, sec))
    current += length

if section_bounds:
    last_start, _, last_sec = section_bounds[-1]
    section_bounds[-1] = (last_start, total_notes, last_sec)

print("\nStructure fine des sections :")
print("-" * 70)
for start, end, sec in section_bounds:
    roles_used = [r for r in sec["roles"] if r in parts]
    print(f"  {sec['name']:16} | {start:3d}→{end:3d} | dens {sec['density']:.2f} | "
          f"tempo {sec['tempo']:3d} | {sec['dynamic']:3s} | "
          f"perc={'oui' if (sec.get('percussion') and HAS_PERCO) else 'non'} | "
          f"rôles={roles_used}")
print("-" * 70)


def fill_part(part, indices, role, instr, section_bounds,
              is_bass=False, is_harmony=False):
    offset = 0.0
    step = 1.0 if is_bass else 0.5

    for start, end, sec in section_bounds:
        if role not in sec["roles"]:
            offset += (end - start) * step
            continue

        density = sec["density"]
        seq = indices[start:end] if not is_bass else indices[start//2 : end//2]
        seq = apply_density(seq, density)

        start_vel = VELOCITY_MAP.get(sec["dynamic"], 70)
        end_vel = VELOCITY_MAP.get(sec.get("crescendo_to", sec["dynamic"]), start_vel)

        n_notes_in_section = len([x for x in seq if x is not None])
        note_count = 0

        for idx in seq:
            if idx is None:
                offset += step
                continue

            # Lissage exponentiel du crescendo (option B)
            if n_notes_in_section > 1:
                progress = note_count / (n_notes_in_section - 1)
                progress = progress ** 1.5
                velocity = int(start_vel + (end_vel - start_vel) * progress)
            else:
                velocity = start_vel
            velocity = max(1, min(127, velocity))

            note_str = int_to_note[idx]
            duration = step * (1.7 if is_bass else 1.0)
            octave_shift = -1 if is_bass else (1 if role == "brass" else 0)

            if is_harmony and np.random.random() < 0.30:
                try:
                    base = make_note_or_chord(note_str, instr,
                                              offset, duration, octave_shift, velocity)
                    if base and isinstance(base, note.Note):
                        third = note.Note(base.pitch.transpose(4))
                        third.storedInstrument = instr
                        third.volume.velocity = velocity
                        c = chord.Chord([base, third])
                        c.offset = offset
                        c.quarterLength = duration
                        part.append(c)
                        offset += step
                        note_count += 1
                        continue
                except Exception:
                    pass

            n = make_note_or_chord(note_str, instr,
                                   offset, duration, octave_shift, velocity)
            if n:
                part.append(n)

            offset += step
            note_count += 1


print("\nRemplissage des pistes...")
for role, part in parts.items():
    if role == "percussion":
        continue
    fill_part(
        part, role_sequences[role], role, DETECTED[role], section_bounds,
        is_bass=(role == "bass"),
        is_harmony=(role == "harmony"),
    )

# ---------- Percussions (UNIQUEMENT si détectées) ----------
if HAS_PERCO:
    print("  → Percussions (instrumentation détectée dans le training)")
    perc_part = parts["percussion"]
    offset = 0.0

    for start, end, sec in section_bounds:
        section_duration = (end - start) * 0.5
        if not sec.get("percussion", False):
            offset += section_duration
            continue

        intensity = sec.get("percussion_intensity", 0.6)
        base_vel = VELOCITY_MAP.get(sec["dynamic"], 80)
        bars = max(1, int(section_duration / 4))

        for bar in range(bars):
            base = offset + bar * 4

            if np.random.random() < intensity:
                kick = note.Note("C2"); kick.storedInstrument = instrument.BassDrum()
                kick.quarterLength = 0.5; kick.offset = base
                kick.volume.velocity = min(127, int(base_vel * 1.1)); perc_part.append(kick)
            if np.random.random() < intensity * 0.9:
                k2 = note.Note("C2"); k2.storedInstrument = instrument.BassDrum()
                k2.quarterLength = 0.5; k2.offset = base + 2
                k2.volume.velocity = min(127, int(base_vel * 1.1)); perc_part.append(k2)
            if np.random.random() < intensity * 0.85:
                sn = note.Note("D2"); sn.storedInstrument = instrument.SnareDrum()
                sn.quarterLength = 0.5; sn.offset = base + 1
                sn.volume.velocity = min(127, int(base_vel)); perc_part.append(sn)
            if np.random.random() < intensity * 0.8:
                sn2 = note.Note("D2"); sn2.storedInstrument = instrument.SnareDrum()
                sn2.quarterLength = 0.5; sn2.offset = base + 3
                sn2.volume.velocity = min(127, int(base_vel)); perc_part.append(sn2)
            if intensity > 0.6 and np.random.random() < 0.35:
                tim = note.Note("A1"); tim.storedInstrument = instrument.Timpani()
                tim.quarterLength = 1.0; tim.offset = base
                tim.volume.velocity = min(127, int(base_vel * 1.2)); perc_part.append(tim)
            if intensity > 0.8 and bar == bars - 1 and np.random.random() < 0.5:
                for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                    f = note.Note("D2"); f.storedInstrument = instrument.SnareDrum()
                    f.quarterLength = 0.25; f.offset = base + t
                    f.volume.velocity = min(127, int(base_vel * 1.15)); perc_part.append(f)

        offset += section_duration
else:
    print("  → Percussions : ABSENTES du training → aucune piste générée")

# ---------- Tempos & dynamiques par section ----------
print("  → Application des tempos et dynamiques par section")

current_offset = 0.0
for start, end, sec in section_bounds:
    section_duration = (end - start) * 0.5

    mm = tempo.MetronomeMark(number=sec["tempo"])
    mm.offset = current_offset
    score.insert(current_offset, mm)

    for role in sec["roles"]:
        if role in parts:
            d = dynamics.Dynamic(sec["dynamic"])
            d.offset = current_offset
            parts[role].insert(current_offset, d)

    if sec.get("percussion", False) and HAS_PERCO:
        d = dynamics.Dynamic(sec["dynamic"])
        d.offset = current_offset
        parts["percussion"].insert(current_offset, d)

    current_offset += section_duration

score.insert(0, tempo.MetronomeMark(number=BASE_TEMPO))
for p in parts.values():
    score.insert(0, p)

# ============================================================
# EXPORT MIDI
# ============================================================
output_midi = os.path.join(OUTPUT_DIR, "generated_epic_orchestra.mid")
score.write('midi', fp=output_midi)
print(f"\nFichier MIDI multipiste généré :")
print(f"  → {output_midi}")

# ============================================================
# SOUNDFONT + RENDU
# ============================================================
print("\n" + "=" * 70)
print("SOUNDFONT & RENDU AUDIO")
print("=" * 70)

soundfont = None
for path in SOUNDFONT_CANDIDATES:
    if os.path.isfile(path):
        soundfont = path
        break

if soundfont:
    print(f"Soundfont trouvé : {soundfont}")
else:
    print("Aucun soundfont standard trouvé → apt install fluidsynth fluid-soundfont-gm")
    soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

output_wav = os.path.join(OUTPUT_DIR, "generated_epic_orchestra.wav")

print("\nCommande manuelle :")
print("-" * 70)
print(f'fluidsynth -ni "{soundfont}" \\')
print(f'           "{output_midi}" \\')
print(f'           -F "{output_wav}" -r 44100 -g 1.0')
print("-" * 70)

print("\nTentative de rendu automatique...")
try:
    rc, out, err = run_cmd(
        ["fluidsynth", "-ni", soundfont, output_midi,
         "-F", output_wav, "-r", "44100", "-g", "1.0"],
        timeout=180,
    )
    if rc == 0 and os.path.exists(output_wav):
        print(f"[OK] Fichier audio généré : {output_wav}")
    else:
        print("[INFO] Rendu automatique échoué. Lancez la commande ci-dessus.")
except FileNotFoundError:
    print("[INFO] fluidsynth non trouvé : sudo apt install fluidsynth fluid-soundfont-gm")
except KeyboardInterrupt:
    print("\n[INFO] Rendu interrompu.")
    raise

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
print(f"MIDI  : {output_midi}")
print(f"WAV   : {output_wav}")
print(f"Fallback registres utilisé : {IS_FALLBACK}")
print("Pistes générées :")
for role, inst in DETECTED.items():
    print(f"  - {role:12} ({type(inst).__name__})")
if not HAS_PERCO:
    print("  - (aucune piste percussion : absente du training)")

_cleanup_children()
