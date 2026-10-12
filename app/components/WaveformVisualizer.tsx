'use client';

import { useRef, useEffect, useCallback } from 'react';

interface WaveformVisualizerProps {
  analyserNode: AnalyserNode | null;
  isActive: boolean;
  rmsLevel: number;
}

export default function WaveformVisualizer({
  analyserNode,
  isActive,
  rmsLevel,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const dataArrayRef = useRef<Uint8Array | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Responsive sizing
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }

    const w = rect.width;
    const h = rect.height;

    // Background
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.06)';
    ctx.lineWidth = 0.5;
    const gridSpacingY = 20;
    for (let y = gridSpacingY; y < h; y += gridSpacingY) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    const gridSpacingX = 40;
    for (let x = gridSpacingX; x < w; x += gridSpacingX) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Label
    ctx.font = '10px ui-monospace, "Cascadia Code", "Fira Code", monospace';
    ctx.fillStyle = 'rgba(100, 200, 255, 0.5)';
    ctx.fillText('LIVE AUDIO WAVEFORM', 8, 14);

    if (!analyserNode || !isActive) {
      // "Waiting" message
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(100, 200, 255, 0.3)';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for audio...', w / 2, h / 2);
      ctx.textAlign = 'start';
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    const bufferLength = analyserNode.fftSize;
    if (!dataArrayRef.current || dataArrayRef.current.length !== bufferLength) {
      dataArrayRef.current = new Uint8Array(bufferLength);
    }
    const dataArray = dataArrayRef.current;
    analyserNode.getByteTimeDomainData(dataArray as any);

    // Color based on rmsLevel
    const intensity = Math.min(1, rmsLevel * 3);
    const r = Math.round(30 + 30 * intensity);
    const g = Math.round(200 + 55 * intensity);
    const b = Math.round(180 + 75 * intensity);
    const lineColor = `rgb(${r}, ${g}, ${b})`;
    const glowColor = `rgba(${r}, ${g}, ${b}, ${0.25 + intensity * 0.35})`;

    // Glow effect
    ctx.shadowBlur = 8 + intensity * 16;
    ctx.shadowColor = glowColor;

    // Draw waveform
    ctx.lineWidth = 1.5 + intensity;
    ctx.strokeStyle = lineColor;
    ctx.beginPath();

    const sliceWidth = w / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    ctx.stroke();

    // Reset shadow
    ctx.shadowBlur = 0;
    ctx.shadowColor = 'transparent';

    animFrameRef.current = requestAnimationFrame(draw);
  }, [analyserNode, isActive, rmsLevel]);

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
        style={{ height: 120 }}
      />
    </div>
  );
}
