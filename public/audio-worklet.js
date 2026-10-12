// AudioWorklet processor for real-time PCM audio capture
// Runs in a separate audio rendering thread
// Collects audio frames and sends them to the main thread

class BharosaAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096; // ~93ms at 44100Hz
    this._buffer = new Float32Array(this._bufferSize);
    this._writeIndex = 0;
    this._isRecording = true;

    this.port.onmessage = (event) => {
      if (event.data.command === 'stop') {
        this._isRecording = false;
      } else if (event.data.command === 'start') {
        this._isRecording = true;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || !this._isRecording) return true;

    const channelData = input[0]; // mono

    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._writeIndex] = channelData[i];
      this._writeIndex++;

      if (this._writeIndex >= this._bufferSize) {
        // Buffer full - send to main thread
        this.port.postMessage({
          type: 'audio-buffer',
          buffer: this._buffer.slice(), // Copy
          sampleRate: sampleRate,
          timestamp: currentTime,
        });
        this._writeIndex = 0;
      }
    }

    // Also send small chunks for visualization
    this.port.postMessage({
      type: 'audio-frame',
      frame: channelData.slice(),
      sampleRate: sampleRate,
      timestamp: currentTime,
    });

    return true;
  }
}

registerProcessor('bharosa-audio-processor', BharosaAudioProcessor);
