import json
import base64
import struct
import soundfile as sf
import time
import os
import threading
import websocket  # websocket-client library
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class RealtimeAudioTranslator:
    def __init__(self, target_language="marathi"):
        self.url = f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        self.target_language = target_language
        self.ws = None
        self.audio_chunks = []
        self.translation_complete = False
        self.session_configured = False
        
        # Timing - CORRECTED
        self.process_start_time = None          # When main process starts
        self.chunk_sending_start_time = None    # When we start sending chunks to OpenAI
        self.chunk_sending_end_time = None      # When all chunks are sent
        self.first_response_time = None         # When we get first translation response
        self.translation_complete_time = None   # When translation is fully complete
        
    def float_to_16bit_pcm(self, float32_array):
        """Convert float32 audio to 16-bit PCM"""
        clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
        pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
        return pcm16

    def base64_encode_audio(self, float32_array):
        """Encode audio data to base64"""
        pcm_bytes = self.float_to_16bit_pcm(float32_array)
        return base64.b64encode(pcm_bytes).decode('ascii')
    
    def on_open(self, ws):
        """Called when WebSocket connection opens"""
        print("✅ Connected to OpenAI Realtime API!")
        
        # Send session configuration
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": f"You are a helpful assistant that translates English speech to {self.target_language}. Respond only with the translated audio in {self.target_language}. Do not provide any explanations or extra commentary.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800
                },
                "temperature": 0.7
            }
        }
        ws.send(json.dumps(session_config))
        print(f"⚙️  Session configured for English → {self.target_language} translation")

    def on_message(self, ws, message):
        """Handle messages from OpenAI server"""
        try:
            event = json.loads(message)
            event_type = event.get("type")
            
            if event_type == "session.created":
                print("🔗 Session created")
                
            elif event_type == "session.updated":
                print("✅ Session updated successfully")
                self.session_configured = True
                
            elif event_type == "input_audio_buffer.speech_started":
                print("🎤 Speech detected in audio")
                
            elif event_type == "input_audio_buffer.speech_stopped":
                print("🔇 Speech ended")
                
            elif event_type == "input_audio_buffer.committed":
                print("📝 Audio buffer committed successfully")
                
            elif event_type == "response.created":
                print("🚀 Translation response started")
                if not self.first_response_time:
                    self.first_response_time = time.time()
                    if self.chunk_sending_start_time:
                        # TRUE latency: from chunk sending start to first response
                        true_latency = self.first_response_time - self.chunk_sending_start_time
                        print(f"⚡ TRUE API Latency: {true_latency:.3f} seconds")
                        
                        # Also show time from chunks completion to response
                        if self.chunk_sending_end_time:
                            response_delay = self.first_response_time - self.chunk_sending_end_time
                            print(f"⏱️  Response delay after chunks sent: {response_delay:.3f} seconds")
                
            elif event_type == "response.output_item.added":
                print("📦 Output item added")
                
            elif event_type == "response.content_part.added":
                print("📄 Content part added")
                
            elif event_type == "response.audio.delta":
                # Collect audio chunks
                audio_data = event.get("delta", "")
                if audio_data:
                    self.audio_chunks.append(audio_data)
                    print(".", end="", flush=True)
                    
            elif event_type == "response.audio.done":
                print(f"\n✅ Translation audio complete! ({len(self.audio_chunks)} chunks)")
                self.translation_complete_time = time.time()
                
                # Calculate actual audio duration for comparison
                if self.audio_chunks:
                    # Estimate duration: each chunk is ~24kHz PCM16
                    total_audio_bytes = sum(len(base64.b64decode(chunk)) for chunk in self.audio_chunks)
                    estimated_duration = total_audio_bytes / (24000 * 2)  # 2 bytes per sample at 24kHz
                    print(f"📏 Estimated output duration: {estimated_duration:.2f} seconds")
                    
                    # Warn if output is significantly shorter than input
                    input_duration = 10.51  # Known input duration
                    if estimated_duration < input_duration * 0.5:
                        print(f"⚠️  OUTPUT SEEMS SHORT! Input: {input_duration:.1f}s, Output: {estimated_duration:.1f}s")
                        print(f"💡 This might indicate incomplete translation for {self.target_language}")
                
                self.save_translation()
                self.print_summary()
                
            elif event_type == "response.done":
                print("🏁 Response completed")
                self.translation_complete = True
                
            elif event_type == "error":
                print(f"❌ API Error: {event}")
                
        except json.JSONDecodeError:
            print("⚠️  Failed to parse server message")
        except Exception as e:
            print(f"❌ Error handling message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔌 Connection closed: {close_status_code} - {close_msg}")

    def send_audio_file(self, audio_file_path):
        """Send audio file in chunks"""
        # Wait for session to be configured
        while not self.session_configured:
            time.sleep(0.1)
            
        try:
            # Load audio file
            data, samplerate = sf.read(audio_file_path, dtype='float32')
            if data.ndim > 1:
                data = data[:, 0]  # Convert to mono
            
            print(f"📂 Loaded audio: {len(data)} samples at {samplerate}Hz ({len(data)/samplerate:.2f}s)")
            
            # Send audio in chunks
            chunk_size = int(samplerate * 0.1)  # 100ms chunks
            total_chunks = (len(data) // chunk_size) + 1
            
            print(f"🚀 Streaming {total_chunks} chunks...")
            self.chunk_sending_start_time = time.time()  # TRUE start time for latency
            
            for i in range(0, len(data), chunk_size):
                if not self.ws or self.ws.sock is None:
                    break
                    
                chunk = data[i:i + chunk_size]
                base64_audio = self.base64_encode_audio(chunk)
                
                # Send chunk to OpenAI
                audio_message = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }
                self.ws.send(json.dumps(audio_message))
                
                # Show progress every 20 chunks
                chunk_num = (i // chunk_size) + 1
                if chunk_num % 20 == 0:
                    elapsed = time.time() - self.chunk_sending_start_time
                    print(f"📤 Sent {chunk_num}/{total_chunks} chunks ({elapsed:.2f}s)")
                
                # Small delay to avoid overwhelming
                time.sleep(0.01)
            
            self.chunk_sending_end_time = time.time()  # When chunk sending completed
            chunks_duration = self.chunk_sending_end_time - self.chunk_sending_start_time
            print(f"✅ All audio chunks sent in {chunks_duration:.3f} seconds!")
            
            # Commit audio and request response
            self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("📝 Audio committed")
            
            # Request response
            response_request = {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "instructions": f"Translate the provided English audio to {self.target_language}. Respond only with the translated audio."
                }
            }
            self.ws.send(json.dumps(response_request))
            print("🎯 Translation requested")
            
        except Exception as e:
            print(f"❌ Error sending audio: {e}")

    def save_translation(self):
        """Save translated audio to WAV file"""
        if not self.audio_chunks:
            print("⚠️  No audio chunks to save")
            return
            
        try:
            # Combine all audio chunks
            combined_audio = "".join(self.audio_chunks)
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(combined_audio)
            
            # Convert PCM16 bytes back to float32
            pcm16_data = struct.unpack('<' + 'h' * (len(audio_bytes) // 2), audio_bytes)
            float32_data = [x / 32767.0 for x in pcm16_data]
            
            # Save to WAV file
            output_filename = f"translated_output_{self.target_language}_{int(time.time())}.wav"
            sf.write(output_filename, float32_data, 24000, format='WAV')
            print(f"🎉 Translation saved: {output_filename}")
            
        except Exception as e:
            print(f"❌ Error saving translation: {e}")

    def print_summary(self):
        """Print CORRECTED timing summary"""
        print("\n" + "="*60)
        print("📊 CORRECTED TRANSLATION PERFORMANCE SUMMARY")
        print("="*60)
        
        # Calculate all timing metrics
        if self.chunk_sending_start_time and self.chunk_sending_end_time:
            chunk_sending_time = self.chunk_sending_end_time - self.chunk_sending_start_time
            print(f"📤 Chunk sending time: {chunk_sending_time:.3f}s")
        
        if self.chunk_sending_start_time and self.first_response_time:
            true_api_latency = self.first_response_time - self.chunk_sending_start_time
            print(f"⚡ TRUE API latency: {true_api_latency:.3f}s")
        
        if self.first_response_time and self.translation_complete_time:
            translation_processing = self.translation_complete_time - self.first_response_time
            print(f"🔄 Translation processing: {translation_processing:.3f}s")
        
        # MOST IMPORTANT: True end-to-end time
        if self.chunk_sending_start_time and self.translation_complete_time:
            TRUE_TOTAL_TIME = self.translation_complete_time - self.chunk_sending_start_time
            print(f"🎯 TRUE TOTAL TIME (chunks→response): {TRUE_TOTAL_TIME:.3f}s")
            print(f"🏃 Realtime factor: {TRUE_TOTAL_TIME/10.51:.2f}x")
            
            if TRUE_TOTAL_TIME < 5:
                print("🚀 EXCELLENT: Sub-5-second processing!")
            elif TRUE_TOTAL_TIME < 10:
                print("✅ GOOD: Near realtime performance")
            else:
                print("⚠️  Could be faster")
        
        print(f"🎵 Audio chunks received: {len(self.audio_chunks)}")
        
        # Language-specific analysis
        chunks_per_second = len(self.audio_chunks) / 10.51 if len(self.audio_chunks) > 0 else 0
        print(f"📈 Output density: {chunks_per_second:.1f} chunks/sec of input")
        
        if chunks_per_second < 2:
            print(f"⚠️  LOW OUTPUT DENSITY - possible incomplete translation for {self.target_language}")
        
        print("="*60)

    def translate_audio(self, audio_file_path):
        """Main translation method"""
        print(f"🎬 Starting translation with OpenAI Realtime API")
        print(f"📁 Audio file: {audio_file_path}")
        print(f"🌍 Target language: {self.target_language}")
        print("="*60)
        
        if not OPENAI_API_KEY:
            print("❌ OPENAI_API_KEY not found. Check your .env file.")
            return
        
        # Record overall process start time
        self.process_start_time = time.time()
        
        # Set up headers for authentication
        headers = [
            f"Authorization: Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta: realtime=v1"
        ]
        
        print("🔌 Connecting to OpenAI Realtime API...")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection
        time.sleep(2)
        
        if self.session_configured:
            # Send audio file in a separate thread
            audio_thread = threading.Thread(target=self.send_audio_file, args=(audio_file_path,))
            audio_thread.start()
            
            # Wait for translation to complete
            timeout = 60  # 60 second timeout
            start_wait = time.time()
            
            while not self.translation_complete and (time.time() - start_wait) < timeout:
                time.sleep(0.5)
            
            if self.translation_complete:
                print("🎉 Translation completed successfully!")
                
                # Final summary with corrected timing
                if self.process_start_time and self.translation_complete_time:
                    overall_time = self.translation_complete_time - self.process_start_time
                    print(f"📊 Overall process time (including setup): {overall_time:.3f}s")
            else:
                print("⚠️  Translation timed out")
                
            audio_thread.join(timeout=5)
        else:
            print("❌ Failed to configure session")
        
        # Close connection
        if self.ws:
            self.ws.close()

def main():
    audio_file = '/Users/hanama/Desktop/AEOS_WORK/labs/speech-translation/english_input.wav'
    target_language = "marathi"
    
    translator = RealtimeAudioTranslator(target_language)
    translator.translate_audio(audio_file)

if __name__ == "__main__":
    main()