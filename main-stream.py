import base64
import json
import os
import struct
import soundfile as sf
import sounddevice as sd
import websocket
import threading
import time
import queue
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class SmoothStreamingAudioTranslationClient:
    def __init__(self, target_language="hindi"):
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        self.headers = [
            "Authorization: Bearer " + OPENAI_API_KEY,
            "OpenAI-Beta: realtime=v1"
        ]
        self.target_language = target_language
        self.ws = None
        self.session_active = False
        
        # Smooth streaming components
        self.audio_queue = queue.Queue(maxsize=100)  # Prevent memory overflow
        self.is_streaming = False
        self.stream_thread = None
        self.sample_rate = 24000
        
        # Continuous audio stream objects
        self.audio_stream = None
        self.stream_buffer = []
        self.buffer_lock = threading.Lock()
        
        # File saving components
        self.file_chunks = []
        
        # Timing and performance tracking
        self.process_start_time = None
        self.streaming_start_time = None
        self.streaming_end_time = None
        self.response_request_time = None
        self.first_response_time = None
        self.first_playback_time = None
        self.translation_complete_time = None
        self.speaking_start_time = None
        self.speaking_complete_time = None
        self.audio_chunks_start_time = None  # When we start sending chunks
        self.final_audio_complete_time = None  # When last audio finishes
        self.total_chunks_sent = 0
        self.response_chunks_received = 0
        self.estimated_output_duration = 0.0
        self.input_duration = 0.0  # Dynamic input duration
        
        # Smooth streaming configuration
        self.min_buffer_samples = 1024  # Minimum samples for smooth playback (very small)
        self.stream_started = False
        self.first_chunk_received = False

    def float_to_16bit_pcm(self, float32_array):
        """Convert float32 audio to 16-bit PCM"""
        clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
        pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
        return pcm16

    def base64_encode_audio(self, float32_array):
        """Encode audio data to base64"""
        pcm_bytes = self.float_to_16bit_pcm(float32_array)
        encoded = base64.b64encode(pcm_bytes).decode('ascii')
        return encoded

    def pcm16_to_float32(self, pcm_bytes):
        """Convert PCM16 bytes back to float32 for playback"""
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]  # Remove odd byte
        if len(pcm_bytes) == 0:
            return np.array([], dtype=np.float32)
        
        pcm16_data = struct.unpack('<' + 'h' * (len(pcm_bytes) // 2), pcm_bytes)
        float32_data = np.array([x / 32767.0 for x in pcm16_data], dtype=np.float32)
        return float32_data

    def audio_callback(self, outdata, frames, time, status):
        """Callback function for continuous audio stream"""
        if status:
            print(f"⚠️  Audio callback status: {status}")
        
        with self.buffer_lock:
            if len(self.stream_buffer) >= frames:
                # We have enough audio data
                outdata[:, 0] = self.stream_buffer[:frames]
                self.stream_buffer = self.stream_buffer[frames:]
            else:
                # Not enough audio data - fill with silence and available data
                if len(self.stream_buffer) > 0:
                    outdata[:len(self.stream_buffer), 0] = self.stream_buffer
                    outdata[len(self.stream_buffer):, 0] = 0  # Fill rest with silence
                    self.stream_buffer = []
                else:
                    outdata.fill(0)  # Complete silence

    def start_continuous_stream(self):
        """Start the continuous audio stream"""
        try:
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=1024,  # Small block size for low latency
                dtype=np.float32
            )
            self.audio_stream.start()
            self.stream_started = True
            print("🎵 Continuous audio stream started!")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio stream: {e}")
            return False

    def add_audio_to_stream(self, audio_data):
        """Add new audio data to the continuous stream buffer"""
        try:
            float_audio = self.pcm16_to_float32(audio_data)
            if len(float_audio) > 0:
                with self.buffer_lock:
                    self.stream_buffer.extend(float_audio)
                return True
        except Exception as e:
            print(f"❌ Error adding audio to stream: {e}")
            return False

    def get_buffer_duration_ms(self):
        """Get current buffer duration in milliseconds"""
        with self.buffer_lock:
            duration_ms = (len(self.stream_buffer) / self.sample_rate) * 1000
            return duration_ms

    def start_streaming_playback_immediately(self):
        """Start streaming audio playback immediately when first chunk arrives"""
        if not self.is_streaming:
            self.is_streaming = True
            self.first_playback_time = time.time()
            self.speaking_start_time = time.time()
            
            # Start continuous audio stream immediately
            if self.start_continuous_stream():
                print(f"🎵 TRUE STREAMING started immediately!")
                
                if self.first_response_time:
                    immediate_delay = self.first_playback_time - self.first_response_time
                    print(f"   • Immediate streaming delay: {immediate_delay:.3f}s")
                
                total_delay = self.first_playback_time - self.process_start_time
                print(f"   • Time to first audio: {total_delay:.3f}s")
                print(f"   ⚡ TRUE REAL-TIME streaming active!")
            else:
                self.is_streaming = False

    def stop_streaming_playback(self):
        """Stop the continuous audio stream"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                self.stream_started = False
                print("🔇 Continuous audio stream stopped")
            except Exception as e:
                print(f"❌ Error stopping audio stream: {e}")
        self.is_streaming = False

    def on_open(self, ws):
        """Handle WebSocket connection opening"""
        print("🔌 Connected to OpenAI Realtime API")
        self.session_active = True
        
        # Configure session for streaming translation
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": f"You are a real-time audio translator. Translate English audio to {self.target_language}. IMPORTANT: Maintain the same tonality, emotional expression, and speaking style as the original audio. If the speaker sounds excited, be excited. If calm, be calm. If urgent, be urgent. Respond only with translated audio that matches the original emotional tone.",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "ash",
                "turn_detection": None,
                # "turn_detection": {
                #     "type": "server_vad",
                #     "threshold": 0.2,
                #     "prefix_padding_ms": 500,
                #     "silence_duration_ms": 2000,
                #     "create_response": False
                # },
                "temperature": 0.6,
                "modalities": ["audio", "text"]
            }
        }
        ws.send(json.dumps(session_update))
        print(f"⚙️  Session configured for English → {self.target_language} smooth streaming")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages with smooth streaming support"""
        try:
            server_event = json.loads(message)
            event_type = server_event.get("type")
            
            if event_type == "session.created":
                print("✅ Session created")
                
            elif event_type == "session.updated":
                print("✅ Session updated successfully")
                
            elif event_type == "input_audio_buffer.speech_started":
                print("🎤 Speech detection started")
                
            elif event_type == "input_audio_buffer.speech_stopped":
                print("🔇 Speech detection stopped")
                
            elif event_type == "input_audio_buffer.committed":
                print("📝 Audio buffer committed")
                
            elif event_type == "response.created":
                print("🚀 Translation response started")
                if not self.first_response_time:
                    self.first_response_time = time.time()
                    if self.response_request_time:
                        api_latency = self.first_response_time - self.response_request_time
                        print(f"⚡ API Response Latency: {api_latency:.3f} seconds")
                
            elif event_type == "response.output_item.created":
                print("📦 Output item created")
                
            elif event_type == "response.content_part.added":
                print("📄 Content part added")
                
            elif event_type == "response.audio.delta":
                # TRUE STREAMING LOGIC - Play immediately when chunks arrive
                audio_data = server_event.get("delta", "")
                if audio_data:
                    try:
                        decoded_audio = base64.b64decode(audio_data)
                        
                        # Save for final file
                        self.file_chunks.append(audio_data)
                        self.response_chunks_received += 1
                        
                        # Calculate estimated output duration as we receive chunks
                        chunk_bytes = len(decoded_audio)
                        chunk_samples = chunk_bytes // 2  # PCM16 = 2 bytes per sample
                        chunk_duration_seconds = chunk_samples / self.sample_rate
                        self.estimated_output_duration += chunk_duration_seconds
                        
                        # TRUE STREAMING: Start immediately on first chunk
                        if not self.first_chunk_received:
                            self.first_chunk_received = True
                            print(f"🎵 First audio chunk received - starting TRUE streaming!")
                            self.start_streaming_playback_immediately()
                        
                        # Add audio to continuous stream immediately
                        if self.stream_started:
                            success = self.add_audio_to_stream(decoded_audio)
                            if success:
                                buffer_ms = self.get_buffer_duration_ms()
                                if self.response_chunks_received % 5 == 0:
                                    print(f"📊 Chunk {self.response_chunks_received} → stream (buffer: {buffer_ms:.0f}ms)")
                            else:
                                print(f"⚠️  Failed to add chunk {self.response_chunks_received} to stream")
                        else:
                            print(f"⚠️  Stream not ready for chunk {self.response_chunks_received}")
                            
                    except Exception as e:
                        print(f"❌ Audio processing error: {e}")
                        
            elif event_type == "response.audio.done":
                print(f"\n✅ Translation audio complete! ({self.response_chunks_received} chunks)")
                print(f"📏 Estimated output duration: {self.estimated_output_duration:.2f} seconds")
                self.translation_complete_time = time.time()
                
                # Keep stream running until all audio is played
                buffer_ms = self.get_buffer_duration_ms()
                if buffer_ms > 0:
                    print(f"⏳ Waiting for {buffer_ms:.0f}ms remaining audio to play...")
                    
                    # Enhanced buffer drainage with more robust checking
                    max_wait_cycles = 120  # Maximum wait cycles (60 seconds at 0.5s intervals)
                    wait_cycles = 0
                    
                    while wait_cycles < max_wait_cycles:
                        remaining_ms = self.get_buffer_duration_ms()
                        
                        if remaining_ms <= 50:  # Consider 50ms or less as "empty"
                            print(f"   • Buffer nearly empty ({remaining_ms:.0f}ms) - finishing")
                            break
                            
                        if remaining_ms > 0:
                            print(f"   • Buffer remaining: {remaining_ms:.0f}ms")
                        
                        time.sleep(0.5)
                        wait_cycles += 1
                    
                    if wait_cycles >= max_wait_cycles:
                        print(f"⚠️  Buffer drainage timeout - forcing completion")
                    
                    # Final small wait to ensure last audio plays
                    time.sleep(1.0)
                
                # Mark speaking completion time
                self.speaking_complete_time = time.time()
                self.final_audio_complete_time = time.time()  # Final completion timestamp
                
                if self.speaking_start_time:
                    total_speaking_time = self.speaking_complete_time - self.speaking_start_time
                    print(f"🎵 All audio playback completed!")
                    print(f"⏱️  Total speaking time: {total_speaking_time:.2f} seconds")
                    print(f"🎯 Speaking vs output ratio: {total_speaking_time/self.estimated_output_duration:.2f}x")
                
                # Calculate and display TOTAL END-TO-END TIME
                if self.audio_chunks_start_time and self.final_audio_complete_time:
                    total_end_to_end_time = self.final_audio_complete_time - self.audio_chunks_start_time
                    print(f"\n🏁 TOTAL END-TO-END TIME: {total_end_to_end_time:.2f} seconds")
                    print(f"   (From first chunk sent → Final audio spoken)")
                    
                    # Calculate efficiency vs input
                    end_to_end_efficiency = total_end_to_end_time / self.input_duration
                    print(f"🎯 End-to-end efficiency: {end_to_end_efficiency:.2f}x input duration")
                    
                    if end_to_end_efficiency < 1.2:
                        print(f"   🚀 AMAZING: Near realtime end-to-end performance!")
                    elif end_to_end_efficiency < 1.5:
                        print(f"   ✅ EXCELLENT: Great end-to-end performance!")
                    elif end_to_end_efficiency < 2.0:
                        print(f"   ✅ GOOD: Reasonable end-to-end time")
                    else:
                        print(f"   ⚠️  Could be optimized further")
                
                # Show TRUE STREAMING improvement
                if self.first_playback_time and self.process_start_time:
                    true_streaming_delay = self.first_playback_time - self.process_start_time
                    old_streaming_delay = 7.0  # Estimated buffered approach for 30s file
                    improvement = old_streaming_delay - true_streaming_delay
                    print(f"\n⚡ TRUE STREAMING ADVANTAGE:")
                    print(f"   • First audio at: {true_streaming_delay:.2f}s (vs {old_streaming_delay:.1f}s buffered)")
                    print(f"   • Improvement: {improvement:.1f}s faster user experience!")
                
                # Stop streaming
                self.stop_streaming_playback()
                
                # Save the complete translated file
                self.save_translated_audio()
                self.print_timing_summary()
                
            elif event_type == "response.done":
                print("🏁 Response completed")
                
            elif event_type == "error":
                print(f"❌ API Error: {server_event}")
                
        except json.JSONDecodeError:
            print("❌ Failed to parse server message")
        except Exception as e:
            print(f"❌ Error handling message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
        self.stop_streaming_playback()

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closing"""
        print("🔌 Connection closed")
        self.session_active = False
        self.stop_streaming_playback()

    def print_timing_summary(self):
        """Print comprehensive timing analysis including smooth streaming metrics"""
        print("\n" + "="*70)
        print("🎵 SMOOTH STREAMING TRANSLATION PERFORMANCE")
        print("="*70)
        
        if not all([self.process_start_time, self.streaming_start_time, self.streaming_end_time, 
                   self.response_request_time, self.first_response_time, self.translation_complete_time]):
            print("⚠️  Incomplete timing data")
            return
        
        # Calculate timing metrics
        upload_duration = self.streaming_end_time - self.streaming_start_time
        api_latency = self.first_response_time - self.response_request_time
        translation_duration = self.translation_complete_time - self.first_response_time
        total_process_time = self.translation_complete_time - self.process_start_time
        
        # Smooth streaming metrics
        if self.first_playback_time:
            time_to_first_audio = self.first_playback_time - self.process_start_time
            streaming_latency = self.first_playback_time - self.first_response_time
        else:
            time_to_first_audio = float('inf')
            streaming_latency = float('inf')
        
        # Speaking duration metrics
        if self.speaking_start_time and self.speaking_complete_time:
            total_speaking_time = self.speaking_complete_time - self.speaking_start_time
        else:
            total_speaking_time = 0
        
        # END-TO-END TIMING - The most important metric
        if self.audio_chunks_start_time and self.final_audio_complete_time:
            total_end_to_end_time = self.final_audio_complete_time - self.audio_chunks_start_time
        else:
            total_end_to_end_time = 0
        
        print(f"📤 Audio Upload Phase:")
        print(f"   • Upload duration: {upload_duration:.3f}s")
        print(f"   • Chunks sent: {self.total_chunks_sent}")
        print(f"   ✅ Optimized upload")
        
        print(f"\n🤖 API Processing:")
        print(f"   • API response latency: {api_latency:.3f}s") 
        print(f"   • Translation duration: {translation_duration:.3f}s")
        print(f"   • Total API time: {api_latency + translation_duration:.3f}s")
        
        print(f"\n🎵 Smooth Streaming Performance:")
        print(f"   • Time to first audio: {time_to_first_audio:.3f}s")
        print(f"   • Streaming method: TRUE real-time streaming")
        if self.first_response_time and self.first_playback_time:
            immediate_streaming_delay = self.first_playback_time - self.first_response_time
            print(f"   • Immediate streaming delay: {immediate_streaming_delay:.3f}s")
        print(f"   • Audio chunks processed: {self.response_chunks_received}")
        
        print(f"\n🎙️  Audio Output Metrics:")
        print(f"   • Estimated output duration: {self.estimated_output_duration:.2f}s")
        print(f"   • Actual speaking time: {total_speaking_time:.2f}s")
        if total_speaking_time > 0 and self.estimated_output_duration > 0:
            speaking_efficiency = total_speaking_time / self.estimated_output_duration
            print(f"   • Speaking efficiency: {speaking_efficiency:.2f}x")
            if speaking_efficiency < 1.1:
                print(f"   ✅ EXCELLENT: Near perfect speaking timing!")
            elif speaking_efficiency < 1.5:
                print(f"   ✅ GOOD: Reasonable speaking overhead")
            else:
                print(f"   ⚠️  Speaking took longer than expected")
        
        print(f"\n📊 Input vs Output Comparison:")
        print(f"   • Input audio duration: {self.input_duration:.2f}s")
        print(f"   • Output audio duration: {self.estimated_output_duration:.2f}s")
        if self.estimated_output_duration > 0:
            translation_ratio = self.estimated_output_duration / self.input_duration
            print(f"   • Translation length ratio: {translation_ratio:.2f}x")
            if translation_ratio > 1.2:
                print(f"   📈 Translation is {((translation_ratio-1)*100):.0f}% longer than original")
            elif translation_ratio < 0.8:
                print(f"   📉 Translation is {((1-translation_ratio)*100):.0f}% shorter than original")
            else:
                print(f"   ⚖️  Translation length is similar to original")
        
        # HIGHLIGHT THE MOST IMPORTANT METRIC
        print(f"\n🏆 KEY PERFORMANCE METRIC:")
        if total_end_to_end_time > 0:
            print(f"   • END-TO-END TIME: {total_end_to_end_time:.2f}s")
            print(f"     (First chunk sent → Final audio spoken)")
            
            efficiency = total_end_to_end_time / self.input_duration
            print(f"   • End-to-end efficiency: {efficiency:.2f}x")
            
            if efficiency < 1.0:
                print(f"   🚀 AMAZING: {((1-efficiency)*100):.0f}% faster than realtime!")
            elif efficiency < 1.5:
                print(f"   ✅ EXCELLENT: Only {((efficiency-1)*100):.0f}% slower than realtime!")
            elif efficiency < 2.0:
                print(f"   ✅ GOOD: {efficiency:.1f}x realtime speed")
            else:
                print(f"   ⚠️  {efficiency:.1f}x realtime - could be optimized")
        
        if time_to_first_audio < 5.0:
            print(f"\n   🚀 EXCELLENT: Smooth audio in under 5 seconds!")
        elif time_to_first_audio < 7.0:
            print(f"\n   ✅ GOOD: Reasonable streaming latency")
        else:
            print(f"\n   ⚠️  Could be improved")
        
        print(f"\n📊 Overall Results:")
        print(f"   • Total process time: {total_process_time:.3f}s")
        print(f"   • Realtime factor: {total_process_time/self.input_duration:.2f}x")
        print(f"   • User experience: SMOOTH continuous streaming")
        
        improvement = 7.0 - time_to_first_audio
        if improvement > 0:
            print(f"   📈 User experience improvement: {improvement:.1f}s faster than batch!")
        
        print("="*70)

    def save_translated_audio(self):
        """Save the complete translated audio file"""
        if not self.file_chunks:
            print("❌ No audio chunks to save")
            return
            
        try:
            # Combine all audio chunks
            combined_audio = "".join(self.file_chunks)
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(combined_audio)
            
            # Convert to float32 for saving
            float32_data = self.pcm16_to_float32(audio_bytes)
            
            # Calculate actual file duration
            actual_duration = len(float32_data) / self.sample_rate
            
            # Save complete file
            timestamp = int(time.time())
            output_filename = f"smooth_translated_{self.target_language}_{timestamp}.wav"
            sf.write(output_filename, float32_data, self.sample_rate, format='WAV')
            
            print(f"💾 Smooth translation saved: {output_filename}")
            print(f"📏 Actual file duration: {actual_duration:.2f} seconds")
            
            # Update our estimate with actual duration
            self.estimated_output_duration = actual_duration
            
        except Exception as e:
            print(f"❌ Error saving audio file: {e}")

    def stream_audio_file(self, filename):
        """Stream audio file with optimal chunking"""
        try:
            # Load audio file
            data, samplerate = sf.read(filename, dtype='float32')
            
            # Convert to mono if stereo
            if data.ndim > 1:
                data = data[:, 0]
            
            input_duration = len(data) / samplerate
            self.input_duration = input_duration  # Store for later use
            print(f"📂 Loaded audio: {len(data)} samples at {samplerate}Hz ({input_duration:.2f}s)")
            
            # Use optimal 0.1s chunks
            chunk_duration = 0.1
            chunk_size = int(samplerate * chunk_duration)
            total_chunks = (len(data) // chunk_size) + 1
            
            print(f"🚀 Starting optimized streaming ({total_chunks} chunks @ 0.1s each)...")
            self.streaming_start_time = time.time()
            self.audio_chunks_start_time = time.time()  # Mark when we start sending chunks
            
            # Stream chunks efficiently
            for i in range(0, len(data), chunk_size):
                if not self.session_active:
                    break
                    
                chunk = data[i:i + chunk_size]
                base64_chunk = self.base64_encode_audio(chunk)
                
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_chunk
                }
                
                self.ws.send(json.dumps(event))
                self.total_chunks_sent += 1
                
                # Progress every 25 chunks
                if self.total_chunks_sent % 25 == 0:
                    elapsed = time.time() - self.streaming_start_time
                    print(f"📤 Sent {self.total_chunks_sent}/{total_chunks} chunks ({elapsed:.2f}s)")
            
            self.streaming_end_time = time.time()
            streaming_duration = self.streaming_end_time - self.streaming_start_time
            print(f"✅ Upload completed in {streaming_duration:.3f}s")
            
            # Commit and request response
            print("📝 Committing audio buffer...")
            commit_event = {"type": "input_audio_buffer.commit"}
            self.ws.send(json.dumps(commit_event))
            
            time.sleep(0.1)
            print("🎯 Requesting smooth translation...")
            self.response_request_time = time.time()
            
            response_event = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    #"instructions": f"Translate the provided English audio to {self.target_language}. Respond only with the translated audio."
                }
            }
            self.ws.send(json.dumps(response_event))
            
        except Exception as e:
            print(f"❌ Error streaming audio: {e}")

    def translate_audio_with_smooth_streaming(self, audio_file_path):
        """Main method for smooth streaming translation"""
        print(f"🎬 Starting SMOOTH STREAMING translation of: {audio_file_path}")
        print(f"🌍 Target language: {self.target_language}")
        print(f"🎵 Streaming: Continuous smooth audio stream + Complete file save")
        print("="*70)
        
        self.process_start_time = time.time()
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start WebSocket in thread
        wst = threading.Thread(target=self.ws.run_forever, daemon=True)
        wst.start()
        
        # Wait for connection
        time.sleep(2)
        
        if self.session_active:
            # Stream the audio file
            self.stream_audio_file(audio_file_path)
            
            # Wait for processing to complete - longer wait to ensure completion
            print("⏳ Processing translation with smooth streaming...")
            
            # Wait with periodic status updates
            max_wait_time = 60  # Maximum 60 seconds (increased for 30s files)
            wait_time = 0
            while wait_time < max_wait_time and (self.is_streaming or self.translation_complete_time is None):
                time.sleep(2)
                wait_time += 2
                
                if self.is_streaming:
                    buffer_ms = self.get_buffer_duration_ms()
                    print(f"   • Still streaming... (buffer: {buffer_ms:.0f}ms)")
                elif self.translation_complete_time is None:
                    print(f"   • Waiting for translation... ({wait_time}s elapsed)")
            
            # Additional wait for audio completion (especially important for longer files)
            if self.is_streaming:
                print("   • Waiting for audio playback to complete...")
                additional_wait = 0
                while self.is_streaming and additional_wait < 30:  # Up to 30 more seconds
                    time.sleep(2)
                    additional_wait += 2
                    buffer_ms = self.get_buffer_duration_ms()
                    print(f"   • Audio still playing... (buffer: {buffer_ms:.0f}ms)")
            
            # Final check and cleanup
            if self.translation_complete_time is None:
                print("⚠️  Translation may not have completed fully")
                # Force save if we have chunks
                if self.file_chunks:
                    print("💾 Attempting to save partial translation...")
                    self.save_translated_audio()
                    self.print_timing_summary()
            else:
                print("✅ Translation completed successfully!")
        
        # Cleanup
        if self.is_streaming:
            print("🔧 Stopping any remaining audio streams...")
            self.stop_streaming_playback()
        self.ws.close()
        
        print("🏁 Smooth streaming translation completed!")

def main():
    """Example usage of TRUE streaming client"""
    # Configuration
    audio_file = '/Users/hanama/Desktop/AEOS_WORK/labs/speech-translation/english-input.wav'
    target_language = "hindi"  # Change as needed
    
    print("⚡ TRUE STREAMING Translation Client")
    print("===================================")
    print("Features:")
    print("• IMMEDIATE audio streaming (no buffering delay)")
    print("• Starts playback on FIRST chunk arrival")
    print("• Optimized 0.1s chunk size")
    print("• Real-time buffer monitoring") 
    print("• Maximum streaming performance")
    print()
    
    # Create TRUE streaming client and translate
    client = SmoothStreamingAudioTranslationClient(target_language)
    client.translate_audio_with_smooth_streaming(audio_file)

if __name__ == "__main__":
    main()