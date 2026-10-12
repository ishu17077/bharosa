'use client';

import { useRef, useEffect, useCallback } from 'react';

interface SpectrogramVisualizerProps {
  analyserNode: AnalyserNode | null;
  isActive: boolean;
}

/** Map a dB value to an RGBA colour on the heat-map scale. */
function dbToColor(db: number): [number, number, number] {
  // Normalize dB to 0-1 range (typical range: -100 to 0 dB)
  const t = Math.max(0, Math.min(1, (db + 100) / 100));

  // dark blue → cyan → green → yellow → red
  if (t < 0.25) {
    const s = t / 0.25;
    return [0, Math.round(s * 80), Math.round(40 + s * 140)];
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return [0, Math.round(80 + s * 175), Math.round(180 - s * 40)];
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return [Math.round(s * 255), Math.round(255 - s * 30), Math.round(140 - s * 100)];
  } else {
    const s = (t - 0.75) / 0.25;
    return [255, Math.round(225 - s * 225), Math.round(40 - s * 40)];
  }
}

export default function SpectrogramVisualizer({
  analyserNode,
  isActive,
}: SpectrogramVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const freqDataRef = useRef<Float32Array | null>(null);
  const imageDataRef = useRef<ImageData | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(rect.width * dpr);
    const ph = Math.round(rect.height * dpr);

    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
      ctx.scale(dpr, dpr);
      imageDataRef.current = null; // reset on resize
    }

    const w = rect.width;
    const h = rect.height;

    if (!analyserNode || !isActive) {
      ctx.fillStyle = '#0a0f1a';
      ctx.fillRect(0, 0, w, h);
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(100, 200, 255, 0.5)';
      ctx.fillText('FREQUENCY SPECTRUM', 8, 14);
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(100, 200, 255, 0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for audio...', w / 2, h / 2);
      ctx.textAlign = 'start';
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    const binCount = analyserNode.frequencyBinCount;
    // Show only up to Nyquist/2
    const usableBins = Math.floor(binCount / 2);

    if (!freqDataRef.current || freqDataRef.current.length !== binCount) {
      freqDataRef.current = new Float32Array(binCount);
    }
    analyserNode.getFloatFrequencyData(freqDataRef.current as any);

    // Scrolling spectrogram via pixel manipulation on the physical canvas
    // Shift existing image left by 1 physical pixel
    const existingImage = ctx.getImageData(dpr, 0, pw - dpr, ph);
    ctx.putImageData(existingImage, 0, 0);

    // Draw new column on the right
    const colWidth = 1; // in CSS pixels
    for (let py = 0; py < h; py++) {
      // Map pixel row to frequency bin (bottom = low freq, top = high freq)
      const freqFraction = 1 - py / h;
      const binIndex = Math.floor(freqFraction * usableBins);
      const db = freqDataRef.current[Math.min(binIndex, usableBins - 1)];
      const [r, g, b] = dbToColor(db);

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(w - colWidth, py, colWidth, 1);
    }

    // Frequency axis labels (left side)
    const sampleRate = analyserNode.context.sampleRate;
    const maxFreq = (sampleRate / 2) / 2; // Nyquist / 2
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, 0, 44, h);
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(180, 220, 255, 0.7)';
    const labelCount = 5;
    for (let i = 0; i <= labelCount; i++) {
      const freq = (maxFreq * i) / labelCount;
      const y = h - (i / labelCount) * h;
      const label = freq >= 1000 ? `${(freq / 1000).toFixed(1)}k` : `${Math.round(freq)}`;
      ctx.fillText(label, 4, y - 2);
    }

    // Title label
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(44, 0, 140, 18);
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(100, 200, 255, 0.5)';
    ctx.fillText('FREQUENCY SPECTRUM', 48, 13);

    animFrameRef.current = requestAnimationFrame(draw);
  }, [analyserNode, isActive]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [draw]);

  return (
    <div className="w-full rounded-lg overflow-hidden border border-white/5">
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: 150 }}
      />
    </div>
  );
}
