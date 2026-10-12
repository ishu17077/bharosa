'use client';

import { useState } from 'react';
import type { CRMAction, FusionResult } from '@/app/lib/types';
import { CRM_ACTION_LABELS } from '@/app/lib/types';

interface StepUpModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: FusionResult;
  actionType: CRMAction;
  onStepUpResolution: (approved: boolean) => void;
}

export default function StepUpModal({
  isOpen,
  onClose,
  result,
  actionType,
  onStepUpResolution,
}: StepUpModalProps) {
  const [status, setStatus] = useState<'pending' | 'approved' | 'rejected'>('pending');
  const [auditHash, setAuditHash] = useState<string>('');

  if (!isOpen) return null;

  const handleApprove = () => {
    // Generate simulated SHA-256 audit record
    const hash = '0x' + Array.from(crypto.getRandomValues(new Uint8Array(20)))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
    setAuditHash(hash);
    setStatus('approved');
    setTimeout(() => {
      onStepUpResolution(true);
      onClose();
      setStatus('pending');
    }, 1500);
  };

  const handleReject = () => {
    const hash = '0x' + Array.from(crypto.getRandomValues(new Uint8Array(20)))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
    setAuditHash(hash);
    setStatus('rejected');
    setTimeout(() => {
      onStepUpResolution(false);
      onClose();
      setStatus('pending');
    }, 1800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fade-slide-in">
      <div className="relative w-full max-w-md rounded-2xl border border-[var(--card-border)] bg-[#0a0f1d] p-6 shadow-2xl">
        {/* Phone Notch & Header */}
        <div className="mx-auto mb-4 h-1.5 w-16 rounded-full bg-slate-700" />
        
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🏦</span>
            <div>
              <h4 className="text-xs font-bold text-white tracking-wide">
                BHAROSA DIGITAL BANKING GATEWAY
              </h4>
              <p className="text-[10px] text-slate-400">
                Out-of-Band Auth Verification (RFC 7865 SIPREC Hook)
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-white text-xs px-2 py-1 rounded"
          >
            ✕
          </button>
        </div>

        {/* Advisory Context Box */}
        <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 p-3.5">
          <div className="flex items-start gap-2.5">
            <span className="text-xl">🚨</span>
            <div>
              <p className="text-xs font-semibold text-red-300">
                Non-Voice Mandate Triggered
              </p>
              <p className="text-[11px] text-red-200/80 mt-0.5">
                Synthetic voice acoustic markers detected on high-risk request:
                <strong className="text-white block mt-0.5">
                  {CRM_ACTION_LABELS[actionType]} (Priya Sharma • AC #...9420)
                </strong>
              </p>
            </div>
          </div>
        </div>

        {/* Simulated Customer Smartphone Notification */}
        <div className="mb-4 rounded-xl border border-slate-700 bg-[#121929] p-4 shadow-inner">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2">
            <span className="flex items-center gap-1">
              <span>📱</span> In-App Push Notification
            </span>
            <span className="font-mono text-[10px]">Just now</span>
          </div>

          <div className="rounded-lg bg-black/40 p-3 border border-slate-800">
            <p className="text-xs font-bold text-white">
              Authorize Account Action
            </p>
            <p className="text-[11px] text-slate-300 mt-1">
              A phone banking agent received a request to execute <span className="text-amber-300 font-semibold">{CRM_ACTION_LABELS[actionType]}</span>. Did you authorize this call?
            </p>
            <div className="mt-2 text-[10px] text-slate-400 flex items-center gap-2">
              <span>Risk Score: <strong className="text-red-400 font-mono">{(result.finalScore * 100).toFixed(0)}%</strong></span>
              <span>•</span>
              <span>Acoustic Flaws: <strong className="text-amber-400 font-mono">{result.acoustic.anomalies.length || 1}</strong></span>
            </div>
          </div>
        </div>

        {/* Resolution Buttons or Status */}
        {status === 'pending' ? (
          <div className="flex flex-col gap-2">
            <button
              onClick={handleApprove}
              className="w-full rounded-xl bg-green-600 hover:bg-green-500 py-2.5 text-xs font-bold text-white transition flex items-center justify-center gap-2 shadow-lg shadow-green-900/30"
            >
              <span>🔐</span> Simulate Customer Biometric Approval
            </button>
            <button
              onClick={handleReject}
              className="w-full rounded-xl bg-red-600/80 hover:bg-red-600 py-2.5 text-xs font-bold text-white transition flex items-center justify-center gap-2"
            >
              <span>🚫</span> Reject & Lock Account (Freeze IMPS/NEFT)
            </button>
          </div>
        ) : status === 'approved' ? (
          <div className="rounded-xl border border-green-500/30 bg-green-500/10 p-3 text-center">
            <span className="text-xl">✅</span>
            <p className="text-xs font-bold text-green-300 mt-1">
              Customer Passkey Verified Out-of-Band
            </p>
            <p className="text-[10px] text-green-200/70 font-mono mt-1 break-all">
              Ledger Hash: {auditHash}
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-center">
            <span className="text-xl">🛡️</span>
            <p className="text-xs font-bold text-red-300 mt-1">
              Fraud Incident Sealed • ₹4,80,000 Preserved
            </p>
            <p className="text-[10px] text-red-200/70 font-mono mt-1 break-all">
              Audit Seal: {auditHash}
            </p>
          </div>
        )}

        {/* Cryptographic Compliance Tag */}
        <div className="mt-4 pt-3 border-t border-slate-800/80 text-[10px] text-slate-500 text-center">
          Secured with ECDSA secp256r1 • Zero Voiceprint Retention • DPDP Act Sec. 8 Fraud Exemption
        </div>
      </div>
    </div>
  );
}

