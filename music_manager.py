"""
Music Manager - Handles background music playback
"""

import pygame
import os


class MusicManager:
    def __init__(self, music_folder='music'):
        """Initialize the music manager"""
        if os.path.isabs(music_folder):
            self.music_folder = music_folder
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.music_folder = os.path.join(base_dir, music_folder)
        self.initialized = False
        self.current_track = None
        self.volume = 0.3  # Default volume (0.0 to 1.0)
        self.sound_effects = {}  # Store loaded sound effects
        self.failed_sound_effects = set()  # Avoid repeated warnings for unsupported SFX files
        self.sfx_volume = 0.7  # Sound effects volume (0.0 to 1.0)
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
            print("✓ Music system initialized")
        except Exception as e:
            print(f"⚠ Could not initialize music system: {e}")
            self.initialized = False
    
    def load_music(self, track_name):
        """Load a music track"""
        if not self.initialized:
            return False
        
        try:
            # Build the full path to the music file
            music_path = os.path.join(self.music_folder, track_name)
            
            if not os.path.exists(music_path):
                print(f"⚠ Music file not found: {music_path}")
                return False
            
            pygame.mixer.music.load(music_path)
            self.current_track = track_name
            pygame.mixer.music.set_volume(self.volume)
            print(f"✓ Music loaded: {track_name}")
            return True
        except Exception as e:
            print(f"⚠ Could not load music: {e}")
            return False
    
    def play(self, track_name=None, loops=-1, fade_ms=1000):
        """
        Play music
        Args:
            track_name: Name of the music file (if None, plays currently loaded track)
            loops: Number of times to loop (-1 for infinite)
            fade_ms: Fade-in duration in milliseconds
        """
        if not self.initialized:
            return False
        
        try:
            # Load new track if specified
            if track_name and track_name != self.current_track:
                if not self.load_music(track_name):
                    return False
            
            # Start playback (RPi pygame builds differ in supported play() signatures)
            if fade_ms > 0:
                try:
                    pygame.mixer.music.play(loops=loops, fade_ms=int(fade_ms))
                except TypeError:
                    # Older pygame: no fade_ms support; fall back to regular play.
                    try:
                        pygame.mixer.music.play(int(loops), 0.0)
                    except TypeError:
                        pygame.mixer.music.play(int(loops))
            else:
                try:
                    pygame.mixer.music.play(loops=loops)
                except TypeError:
                    try:
                        pygame.mixer.music.play(int(loops), 0.0)
                    except TypeError:
                        pygame.mixer.music.play(int(loops))
            
            print(f"♪ Playing: {self.current_track}")
            return True
        except Exception as e:
            print(f"⚠ Could not play music: {e}")
            return False
    
    def stop(self, fade_ms=1000):
        """
        Stop music playback
        Args:
            fade_ms: Fade-out duration in milliseconds
        """
        if not self.initialized:
            return
        
        try:
            if fade_ms > 0:
                pygame.mixer.music.fadeout(fade_ms)
            else:
                pygame.mixer.music.stop()
            print("⏹ Music stopped")
        except Exception as e:
            print(f"⚠ Could not stop music: {e}")
    
    def pause(self):
        """Pause music playback"""
        if not self.initialized:
            return
        
        try:
            pygame.mixer.music.pause()
            print("⏸ Music paused")
        except Exception as e:
            print(f"⚠ Could not pause music: {e}")
    
    def unpause(self):
        """Resume music playback"""
        if not self.initialized:
            return
        
        try:
            pygame.mixer.music.unpause()
            print("▶ Music resumed")
        except Exception as e:
            print(f"⚠ Could not resume music: {e}")
    
    def set_volume(self, volume):
        """
        Set music volume
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self.initialized:
            return
        
        try:
            self.volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
            pygame.mixer.music.set_volume(self.volume)
            print(f"🔊 Volume: {int(self.volume * 100)}%")
        except Exception as e:
            print(f"⚠ Could not set volume: {e}")
    
    def is_playing(self):
        """Check if music is currently playing"""
        if not self.initialized:
            return False
        
        try:
            return pygame.mixer.music.get_busy()
        except:
            return False
    
    def load_sound_effect(self, sfx_name):
        """
        Load a sound effect
        Args:
            sfx_name: Name of the sound effect file
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.initialized:
            return False
        
        try:
            # Check if already loaded
            if sfx_name in self.sound_effects:
                return True
            if sfx_name in self.failed_sound_effects:
                return False
            
            # Build the full path to the sound effect file
            primary_path = os.path.join(self.music_folder, sfx_name)
            base_name, ext = os.path.splitext(sfx_name)
            candidate_names = [sfx_name]
            if ext.lower() == '.mp3':
                candidate_names.extend([f"{base_name}.ogg", f"{base_name}.wav"])
            elif ext == '':
                candidate_names.extend([f"{sfx_name}.ogg", f"{sfx_name}.wav", f"{sfx_name}.mp3"])

            candidate_paths = [os.path.join(self.music_folder, name) for name in candidate_names]
            existing_paths = [path for path in candidate_paths if os.path.exists(path)]

            if not existing_paths:
                print(f"⚠ Sound effect file not found: {primary_path}")
                self.failed_sound_effects.add(sfx_name)
                return False
            
            # Load the first decodable candidate (mp3 can fail on some RPi builds).
            last_error = None
            for sfx_path in existing_paths:
                try:
                    sound = pygame.mixer.Sound(sfx_path)
                    sound.set_volume(self.sfx_volume)
                    self.sound_effects[sfx_name] = sound
                    print(f"✓ Sound effect loaded: {os.path.basename(sfx_path)}")
                    return True
                except Exception as e:
                    last_error = e

            # If loading failed (commonly MP3 unsupported on some RPi builds),
            # try converting an MP3 to WAV using ffmpeg/avconv if available.
            try:
                import shutil
                converter = shutil.which('ffmpeg') or shutil.which('avconv')
                if converter:
                    # Prefer first existing mp3 candidate
                    mp3_candidates = [p for p in existing_paths if p.lower().endswith('.mp3')]
                    if mp3_candidates:
                        src = mp3_candidates[0]
                        cache_dir = os.path.join(self.music_folder, '.cache')
                        os.makedirs(cache_dir, exist_ok=True)
                        dst = os.path.join(cache_dir, os.path.splitext(os.path.basename(src))[0] + '.wav')
                        if (not os.path.exists(dst)) or (os.path.getmtime(dst) < os.path.getmtime(src)):
                            cmd = [converter, '-y', '-i', src, dst]
                            try:
                                subprocess_run = __import__('subprocess').run
                                subprocess_run(cmd, check=True, stdout=__import__('subprocess').PIPE, stderr=__import__('subprocess').PIPE)
                            except Exception as e:
                                last_error = e
                                raise

                        try:
                            sound = pygame.mixer.Sound(dst)
                            sound.set_volume(self.sfx_volume)
                            self.sound_effects[sfx_name] = sound
                            print(f"✓ Sound effect loaded (converted): {os.path.basename(dst)}")
                            return True
                        except Exception as e:
                            last_error = e
            except Exception:
                pass

            self.failed_sound_effects.add(sfx_name)
            print(f"⚠ Could not load sound effect: {last_error}")
            return False
        except Exception as e:
            self.failed_sound_effects.add(sfx_name)
            print(f"⚠ Could not load sound effect: {e}")
            return False
    
    def play_sound_effect(self, sfx_name):
        """
        Play a sound effect
        Args:
            sfx_name: Name of the sound effect file
        """
        if not self.initialized:
            return
        
        try:
            # Load if not already loaded
            if sfx_name not in self.sound_effects:
                if not self.load_sound_effect(sfx_name):
                    return
            
            # Play the sound effect
            self.sound_effects[sfx_name].play()
        except Exception as e:
            print(f"⚠ Could not play sound effect: {e}")
    
    def set_sfx_volume(self, volume):
        """
        Set sound effects volume
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self.initialized:
            return
        
        try:
            self.sfx_volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
            # Update volume for all loaded sound effects
            for sound in self.sound_effects.values():
                sound.set_volume(self.sfx_volume)
            print(f"🔊 SFX Volume: {int(self.sfx_volume * 100)}%")
        except Exception as e:
            print(f"⚠ Could not set SFX volume: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                print("✓ Music system cleaned up")
            except Exception as e:
                print(f"⚠ Error during cleanup: {e}")


# Global music manager instance
_music_manager = None


def get_music_manager(music_folder='music'):
    """Get or create the global music manager instance"""
    global _music_manager
    if _music_manager is None:
        _music_manager = MusicManager(music_folder)
    return _music_manager
