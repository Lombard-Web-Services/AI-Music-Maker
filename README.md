# AI-Music-Maker

**by Lombard Web Services**  
*Des solutions innovantes pour un avenir plus intelligent.*

![AI-Music-Maker](image.png)

AI-Music-Maker est un moteur de génération musicale basé sur l'intelligence artificielle. Il utilise un modèle LSTM (Long Short-Term Memory) pour composer des morceaux orchestraux épiques à partir de vos propres fichiers audio (MP3) ou fichiers MIDI. Le système analyse les notes, apprend le style musical, puis génère une partition multipiste avec une structure dramatique avancée (Intro, Build-up, Climax, Outro) et une orchestration adaptative.

---

## Fonctionnalités

- **Transcription Audio vers MIDI** : Utilise Basic Pitch de Spotify pour extraire les notes de vos fichiers MP3.
- **Génération Multi-Pistes** : Crée des partitions pour plusieurs instruments simultanément.
- **Orchestration Adaptative** : Détecte automatiquement les instruments présents dans votre matériel source (violon, piano, clavecin, flûte, orgue, etc.) et s'adapte en conséquence.
- **Structure Épique** : Génère un arc dramatique complet avec 7 sections distinctes (Intro, Awakening, Build-up 1, Build-up 2, Climax, Peak / Release, Outro), avec des changements de tempo et de dynamique progressifs.
- **Rendu Audio Haute Qualité** : Transforme le MIDI généré en fichier WAV grâce à FluidSynth et des banques de sons (SoundFonts).

---

## Prérequis et Installation

### 1. Environnement Python

Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances de ce projet.

```bash
# Créer un environnement virtuel
python3 -m venv venv

# L'activer (Linux / Mac)
source venv/bin/activate

# L'activer (Windows)
venv\Scripts\activate
```

### 2. Dépendances Python

Une fois le venv activé (vous verrez `(venv)` au début de votre ligne de commande), installez les paquets requis :

```bash
pip install --upgrade pip
pip install basic-pitch music21 tensorflow numpy matplotlib onnxruntime
```

**Note pour les utilisateurs de Python 3.11+** : Si `basic-pitch` rencontre des difficultés pour s'installer, utilisez la commande suivante pour forcer l'installation des dépendances TensorFlow :

```bash
pip install basic-pitch[tf] music21 numpy matplotlib
```

### 3. Vérification de l'installation

Testez que les modules principaux sont correctement installés :

```bash
python3 -c "import numpy; print('numpy OK')"
python3 -c "import basic_pitch; print('basic_pitch OK')"
python3 -c "import music21; print('music21 OK')"
```

### 4. FluidSynth (pour le rendu audio)

Pour convertir les fichiers MIDI générés en WAV, vous avez besoin de FluidSynth et d'un SoundFont (banque de sons).

**Sur Ubuntu / Debian :**

```bash
sudo apt update
sudo apt install fluidsynth fluid-soundfont-gm
```

**Sur macOS :**

```bash
brew install fluid-synth
```

**Sur Windows :**  
Téléchargez l'installeur depuis [FluidSynth.org](https://www.fluidsynth.org/) et ajoutez-le à votre PATH.

---

## Utilisation

### Étape 1 : Placer les fichiers sources

Placez vos fichiers audio `.mp3` dans le dossier `solo/`. Le script se chargera de les convertir en MIDI.

*(Si vous disposez déjà de fichiers MIDI, placez-les directement dans le dossier `midi_output/`.)*

### Étape 2 : Lancer la génération

Exécutez le script principal :

```bash
python3 training.py
```

Le script va successivement :

1. Convertir les MP3 en MIDI (si nécessaire).
2. Analyser le contenu pour détecter les instruments.
3. Entraîner (ou charger) le modèle LSTM sur les séquences musicales.
4. Générer une nouvelle composition.
5. Créer un fichier MIDI multipiste (`output/generated_epic_orchestra.mid`).
6. Tenter un rendu audio automatique vers `output/generated_epic_orchestra.wav`.

### Étape 3 : Rendu manuel avec FluidSynth (Optionnel)

Si le rendu automatique échoue, vous pouvez le faire manuellement avec FluidSynth :

```bash
fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 output/generated_epic_orchestra.mid -F output/generated_epic_orchestra.wav -r 44100 -g 1.0
```

---

## Dépannage (Troubleshooting)

### Problème : `basic_pitch` ne s'installe pas ou lève une erreur

- **Solution 1** : Assurez-vous que votre `pip` est à jour.
- **Solution 2** : Si l'erreur persiste, essayez :
  ```bash
  pip install basic-pitch[tf] music21 numpy matplotlib
  ```
- **Solution 3** : Réinstallez toutes les dépendances avec `onnxruntime` :
  ```bash
  pip install --upgrade pip && pip install basic-pitch music21 tensorflow numpy matplotlib onnxruntime
  ```

### Problème : Aucun son n'est généré

- Vérifiez que `fluidsynth` est bien installé (`fluidsynth --version`).
- Vérifiez que le chemin vers le SoundFont est correct dans le script (ou dans la commande manuelle).

### Problème : Tous les instruments sonnent comme un violon

Ce problème survenait avec les anciennes versions de music21. Assurez-vous d'utiliser la version du script `gen-solo-instruments.py` fournie, qui force la réassignation des canaux MIDI via `mido`.

---

## Licence & Copyright

Copyright © 2026 Lombard Web Services. Tous droits réservés.
