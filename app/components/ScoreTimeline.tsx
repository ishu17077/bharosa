'use client';

import { useRef, useEffect, useCallback } from 'react';
import type { ScoreHistoryEntry } from '@/app/lib/types';

interface ScoreTimelineProps {
  history: ScoreHistoryEntry[];
  maxPoints?: number;
}

const LEGEND: { key: 'acousticScore' | 'timingScore' | 'contextScore'; label: string; color: string }[] = [
  { key: 'acousticScore', label: 'Acoustic', color: '#22d3ee' },
  { key: 'timingScore', label: 'Timing', color: '#facc15' },
  { key: 'contextScore', label: 'Context', color: '#a78bfa' },
];

export default function ScoreTimeline({
  history,
  maxPoints = 100,
}: ScoreTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  const paint = useCallback(() => {
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
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 22, right: 12, bottom: 24, left: 36 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Background
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, w, h);

    // Risk zones (bottom-to-top: green, amber, red)
    const zones: { y0: number; y1: number; color: string }[] = [
      { y0: 0.65, y1: 1.0, color: 'rgba(220, 38, 38, 0.08)' },   // RED
      { y0: 0.3, y1: 0.65, color: 'rgba(234, 179, 8, 0.06)' },   // AMBER
      { y0: 0.0, y1: 0.3, color: 'rgba(34, 197, 94, 0.06)' },    // GREEN
    ];
    for (const z of zones) {
      const zy = pad.top + (1 - z.y1) * plotH;
      const zh = (z.y1 - z.y0) * plotH;
      ctx.fillStyle = z.color;
      ctx.fillRect(pad.left, zy, plotW, zh);
    }

    // Grid & Y-axis labels
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(180, 220, 255, 0.4)';
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.06)';
    ctx.lineWidth = 0.5;
    for (let v = 0; v <= 1; v += 0.2) {
      const y = pad.top + (1 - v) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(1), 4, y + 3);
    }

    // Title
    ctx.fillStyle = 'rgba(100, 200, 255, 0.5)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('SCORE TIMELINE', pad.left + 4, 14);

    // Slice data to maxPoints
    const data = history.slice(-maxPoints);
    if (data.length < 2) {
      ctx.fillStyle = 'rgba(100, 200, 255, 0.25)';
      ctx.font = '13px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Collecting data...', w / 2, h / 2);
      ctx.textAlign = 'start';
      return;
    }

    const t0 = data[0].timestamp;
    const tMax = data[data.length - 1].timestamp;
    const tRange = tMax - t0 || 1;

    const toX = (t: number) => pad.left + ((t - t0) / tRange) * plotW;
    const toY = (v: number) => pad.top + (1 - Math.min(1, Math.max(0, v))) * plotH;

    // X-axis labels (relative seconds)
    ctx.fillStyle = 'rgba(180, 220, 255, 0.35)';
    ctx.font = '9px ui-monospace, monospace';
    const xLabels = 5;
    for (let i = 0; i <= xLabels; i++) {
      const t = t0 + (tRange * i) / xLabels;
      const x = toX(t);
      const secs = ((t - t0) / 1000).toFixed(0);
      ctx.fillText(`${secs}s`, x, h - 6);
    }

    // Final score filled area
    ctx.beginPath();
    ctx.moveTo(toX(data[0].timestamp), toY(0));
    for (const pt of data) {
      ctx.lineTo(toX(pt.timestamp), toY(pt.finalScore));
    }
    ctx.lineTo(toX(data[data.length - 1].timestamp), toY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fill();

    // Signal lines
    for (const { key, color } of LEGEND) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < data.length; i++) {
        const x = toX(data[i].timestamp);
        const y = toY(data[i][key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Final score line (white)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    for (let i = 0; i < data.length; i++) {
      const x = toX(data[i].timestamp);
      const y = toY(data[i].finalScore);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend (top-right)
    const legendX = w - pad.right - 120;
    const legendY = pad.top + 4;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(legendX - 6, legendY - 4, 126, 60);
    ctx.font = '9px ui-monospace, monospace';
    const allItems = [...LEGEND, { key: 'finalScore' as const, label: 'Final', color: 'rgba(255,255,255,0.6)' }];
    allItems.forEach((item, i) => {
      const ly = legendY + i * 14;
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, ly, 10, 8);
      ctx.fillStyle = 'rgba(220,230,255,0.7)';
      ctx.fillText(item.label, legendX + 16, ly + 8);
    });
  }, [history, maxPoints]);

  useEffect(() => {
    paint();
    // no rAF loop – repaint on data change only
    return () => cancelAnimationFrame(rafRef.current);
  }, [paint]);

  return (
    <div className="w-full rounded-lg overflow-hidden border border-white/5">
      <canvas
        ref={canvasRef}
        className="w-full block"
        style={{ height: 200 }}
      />
    </div>
  );
}
