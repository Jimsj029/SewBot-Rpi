# Music Folder

This folder contains background music files for the SewBot application.

## Required Music Files

Place the following music files in this directory:

1. **main_menu.mp3** or **main_menu.ogg** - Background music for the main menu
2. **tutorial.mp3** or **tutorial.ogg** - Background music for the sewing setup tutorial
3. **mode_selection.mp3** or **mode_selection.ogg** - Background music for the mode selection screen
4. **wallet.mp3** or **wallet.ogg** - Background music for the wallet tutorial
5. **pattern.mp3** or **pattern.ogg** - Background music for pattern mode levels

## Supported Formats

- MP3 (.mp3)
- OGG Vorbis (.ogg)
- WAV (.wav)

## Notes

- The music system will gracefully handle missing files by displaying a warning
- Music will loop continuously in each scene
- Default volume is set to 50% and can be adjusted using the mute button
- A mute button is available in the upper right corner of all screens
- For best performance on Raspberry Pi, use compressed formats like OGG or MP3

## Adding Your Own Music

1. Place your music files in this folder
2. Name them according to the convention above OR
3. Update the music file names in `main.py` and `pattern_mode.py`

## Mute Button

- Located in the upper right corner of the screen
- Click to mute/unmute all music
- Shows a speaker icon when active, crossed-out speaker when muted
- Mute state persists across all screens

## Placeholder Files

If you don't have music files yet, the application will run without them. The music system will simply report that files are not found but won't crash the application.
