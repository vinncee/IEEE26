"use client";

import React from "react"

import Link from "next/link";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { Navbar } from "@/components/navbar";
import { useScrollReveal } from "@/hooks/use-scroll-reveal";
import { cn } from "@/lib/utils";

/* ============================
   Parallax background component
   ============================ */
function ParallaxBg() {
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          setOffset(window.scrollY);
          ticking = false;
        });
        ticking = true;
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      {/* Large teal circle */}
      <div
        className="absolute -right-32 -top-32 h-[500px] w-[500px] rounded-full bg-primary/[0.06]"
        style={{ transform: `translateY(${offset * 0.12}px)` }}
      />
      {/* Small amber accent */}
      <div
        className="absolute -left-16 top-[60%] h-[300px] w-[300px] rounded-full bg-accent/[0.07]"
        style={{ transform: `translateY(${offset * -0.08}px)` }}
      />
    </div>
  );
}

/* ============================
   Animated counter for stats
   ============================ */
function AnimatedCounter({ end, suffix = "", duration = 2000 }: { end: number; suffix?: string; duration?: number }) {
  const { ref, isVisible } = useScrollReveal<HTMLSpanElement>({ threshold: 0.5 });
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!isVisible) return;
    let start = 0;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      start = Math.round(eased * end);
      setCount(start);
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [isVisible, end, duration]);

  return (
    <span ref={ref} className="tabular-nums">
      {count}{suffix}
    </span>
  );
}

/* ============================
   Hero Section
   ============================ */
function HeroSection() {
  const { ref: headingRef, isVisible: headingVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.1 });
  const { ref: imageRef, isVisible: imageVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.1 });

  return (
    <section
      className="relative flex flex-col items-center gap-8 px-6 py-20 md:py-28 lg:flex-row lg:gap-16 lg:px-16"
      aria-labelledby="hero-heading"
    >
      <ParallaxBg />

      <div
        ref={headingRef}
        className={cn(
          "relative z-10 flex max-w-xl flex-col gap-6 text-center lg:text-left",
          headingVisible ? "animate-slide-in-left" : "opacity-0",
        )}
      >
        <div className="inline-flex self-center rounded-full border border-primary/20 bg-primary/10 px-4 py-1.5 lg:self-start">
          <span className="text-sm font-semibold text-primary">
            Accessible Communication
          </span>
        </div>
        <h1
          id="hero-heading"
          className="text-balance text-4xl font-bold leading-tight tracking-tight text-foreground md:text-5xl lg:text-6xl"
        >
          Breaking barriers between{" "}
          <span className="bg-gradient-to-r from-primary to-teal-500 bg-clip-text text-transparent">
            sign language
          </span>{" "}
          and text
        </h1>
        <p className="text-pretty text-lg leading-relaxed text-muted-foreground md:text-xl">
          SignBridge translates sign language into readable text in real-time
          using your camera. No interpreters needed, no delays -- just point
          your camera, sign, and read.
        </p>
        <div className="flex flex-col items-center gap-3 sm:flex-row lg:items-start">
          <Link
            href="/translate"
            className="group inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-8 py-4 text-lg font-semibold text-primary-foreground no-underline shadow-lg shadow-primary/25 transition-all duration-300 hover:-translate-y-0.5 hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/30 active:translate-y-0"
          >
            <svg
              className="h-5 w-5 transition-transform duration-300 group-hover:scale-110"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="8" />
            </svg>
            Start Translating
          </Link>
          <a
            href="#how-it-works"
            className="inline-flex items-center justify-center rounded-xl border-2 border-border bg-transparent px-8 py-4 text-lg font-semibold text-foreground no-underline transition-all duration-300 hover:-translate-y-0.5 hover:border-primary/40 hover:bg-primary/5"
          >
            Learn More
          </a>
        </div>
      </div>

      <div
        ref={imageRef}
        className={cn(
          "relative z-10 w-full max-w-lg lg:max-w-xl",
          imageVisible ? "animate-slide-in-right" : "opacity-0",
        )}
      >
        <div className="relative">
          <div className="rounded-2xl shadow-2xl shadow-foreground/10 transition-transform duration-500 hover:scale-[1.02] bg-muted h-96 flex items-center justify-center">
            <p className="text-muted-foreground">Sign language translation demo</p>
          </div>
          {/* Floating badge */}
          <div className="absolute -bottom-4 -left-4 animate-float rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100">
                <svg className="h-4 w-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium text-muted-foreground">Accuracy</p>
                <p className="text-sm font-bold text-foreground">Real-time</p>
              </div>
            </div>
          </div>
          {/* Floating badge right */}
          <div className="absolute -right-4 -top-4 animate-float rounded-xl border border-border bg-card px-4 py-3 shadow-lg" style={{ animationDelay: "1.5s" }}>
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
                </svg>
              </div>
              <div>
                <p className="text-xs font-medium text-muted-foreground">Accessible</p>
                <p className="text-sm font-bold text-foreground">WCAG AA+</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================
   Stats Section
   ============================ */
function StatsSection() {
  const { ref, isVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.2 });

  const stats = [
    { value: 466, suffix: "M+", label: "Deaf & hard-of-hearing people worldwide" },
    { value: 300, suffix: "+", label: "Sign languages globally" },
    { value: 2, suffix: "s", label: "Translation latency target" },
    { value: 100, suffix: "%", label: "Browser-based, no install" },
  ];

  return (
    <section className="relative border-y border-border bg-primary/[0.03] px-6 py-12">
      <div
        ref={ref}
        className={cn(
          "mx-auto grid max-w-5xl gap-8 sm:grid-cols-2 lg:grid-cols-4",
          isVisible ? "animate-fade-up" : "opacity-0",
        )}
      >
        {stats.map((stat) => (
          <div key={stat.label} className="flex flex-col items-center text-center">
            <p className="text-3xl font-bold text-primary md:text-4xl">
              <AnimatedCounter end={stat.value} suffix={stat.suffix} />
            </p>
            <p className="mt-1 text-sm text-muted-foreground leading-relaxed">{stat.label}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ============================
   Features Section
   ============================ */
function FeaturesSection() {
  const features = [
    {
      title: "Real-Time Translation",
      description:
        "Hand signs are detected and translated into text with under 2 seconds of delay, keeping conversations flowing naturally.",
      icon: (
        <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
        </svg>
      ),
      color: "from-amber-500/10 to-amber-500/5 text-amber-600",
      iconBg: "bg-amber-100",
    },
    {
      title: "Built for Accessibility",
      description:
        "Large touch targets, high-contrast captions, adjustable text sizes, and full keyboard navigation -- designed with WCAG guidelines in mind.",
      icon: (
        <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" />
        </svg>
      ),
      color: "from-primary/10 to-primary/5 text-primary",
      iconBg: "bg-primary/10",
    },
    {
      title: "Confidence Indicators",
      description:
        "Clear visual indicators show how confident the system is in each translation, so you always know the reliability of the output.",
      icon: (
        <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: "from-emerald-500/10 to-emerald-500/5 text-emerald-600",
      iconBg: "bg-emerald-100",
    },
    {
      title: "Portable and Affordable",
      description:
        "Runs on any device with a camera and browser. No expensive hardware or software required -- ideal for care centers and families.",
      icon: (
        <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3" />
        </svg>
      ),
      color: "from-sky-500/10 to-sky-500/5 text-sky-600",
      iconBg: "bg-sky-100",
    },
  ];

  return (
    <section
      className="px-6 py-16 md:py-24"
      aria-labelledby="features-heading"
    >
      <div className="mx-auto max-w-6xl">
        <SectionHeader
          id="features-heading"
          title="Designed for Everyone"
          subtitle="Whether you are a deaf user, a family member, a volunteer, or a care center worker, SignBridge makes communication effortless."
        />

        <div className="grid gap-6 md:grid-cols-2">
          {features.map((feature, i) => (
            <FeatureCard key={feature.title} feature={feature} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}

function FeatureCard({
  feature,
  index,
}: {
  feature: { title: string; description: string; icon: React.ReactNode; color: string; iconBg: string };
  index: number;
}) {
  const { ref, isVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.15 });

  return (
    <div
      ref={ref}
      className={cn(
        "group relative overflow-hidden rounded-2xl border border-border bg-card p-6 transition-all duration-500 hover:-translate-y-1 hover:shadow-lg hover:shadow-primary/5",
        isVisible ? "animate-fade-up" : "opacity-0",
      )}
      style={{ animationDelay: isVisible ? `${index * 100}ms` : "0ms" }}
    >
      {/* Gradient background on hover */}
      <div className={cn("absolute inset-0 bg-gradient-to-br opacity-0 transition-opacity duration-500 group-hover:opacity-100", feature.color)} />

      <div className="relative z-10">
        <div className={cn("mb-4 flex h-12 w-12 items-center justify-center rounded-xl transition-transform duration-300 group-hover:scale-110", feature.iconBg)}>
          <span className={feature.color.split(" ").pop()}>{feature.icon}</span>
        </div>
        <h3 className="mb-2 text-xl font-bold text-foreground">
          {feature.title}
        </h3>
        <p className="text-base leading-relaxed text-muted-foreground">
          {feature.description}
        </p>
      </div>
    </div>
  );
}

/* ============================
   Section header helper
   ============================ */
function SectionHeader({ id, title, subtitle }: { id: string; title: string; subtitle: string }) {
  const { ref, isVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.3 });

  return (
    <div
      ref={ref}
      className={cn("mb-12 text-center", isVisible ? "animate-fade-up" : "opacity-0")}
    >
      <h2 id={id} className="text-balance text-3xl font-bold text-foreground md:text-4xl">
        {title}
      </h2>
      <p className="mx-auto mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
        {subtitle}
      </p>
    </div>
  );
}

/* ============================
   How It Works Section
   ============================ */
function HowItWorksSection() {
  const steps = [
    { number: "1", title: "Open SignBridge", description: "Launch the app in your browser -- no installation needed.", icon: "M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" },
    { number: "2", title: "Press Start", description: "Allow camera access and begin recording your signs.", icon: "M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" },
    { number: "3", title: "Sign Naturally", description: "Use sign language as you normally would. The camera captures your gestures in real-time.", icon: "M10.05 4.575a1.575 1.575 0 10-3.15 0v3.15a1.575 1.575 0 003.15 0v-3.15zm7.875 0a1.575 1.575 0 10-3.15 0v3.15a1.575 1.575 0 003.15 0v-3.15zM6.9 12.525a1.575 1.575 0 10-3.15 0v3.15a1.575 1.575 0 003.15 0v-3.15zm7.875 0a1.575 1.575 0 10-3.15 0v3.15a1.575 1.575 0 003.15 0v-3.15z" },
    { number: "4", title: "Read Captions", description: "Your signs are translated to text captions instantly, displayed right below the camera feed.", icon: "M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" },
  ];

  return (
    <section
      id="how-it-works"
      className="relative overflow-hidden bg-secondary/50 px-6 py-16 md:py-24"
      aria-labelledby="how-heading"
    >
      <div className="mx-auto max-w-4xl">
        <SectionHeader
          id="how-heading"
          title="How It Works"
          subtitle="Four simple steps to start communicating."
        />

        <div className="relative">
          {/* Connecting line */}
          <div className="absolute left-1/2 top-0 hidden h-full w-px -translate-x-1/2 bg-gradient-to-b from-primary/30 via-primary/20 to-transparent lg:block" aria-hidden="true" />

          <div className="grid gap-8 lg:grid-cols-4">
            {steps.map((step, i) => (
              <StepCard key={step.number} step={step} index={i} />
            ))}
          </div>
        </div>

        <div className="mt-14 text-center">
          <CtaReveal />
        </div>
      </div>
    </section>
  );
}

function StepCard({ step, index }: { step: { number: string; title: string; description: string; icon: string }; index: number }) {
  const { ref, isVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.2 });

  return (
    <div
      ref={ref}
      className={cn(
        "flex flex-col items-center text-center",
        isVisible ? "animate-fade-up" : "opacity-0",
      )}
      style={{ animationDelay: isVisible ? `${index * 150}ms` : "0ms" }}
    >
      <div className="group relative mb-4">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/20 transition-transform duration-300 hover:scale-110 hover:-rotate-3">
          <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d={step.icon} />
          </svg>
        </div>
        <span className="absolute -right-2 -top-2 flex h-7 w-7 items-center justify-center rounded-full bg-accent text-xs font-bold text-accent-foreground shadow-md">
          {step.number}
        </span>
      </div>
      <h3 className="mb-2 text-lg font-bold text-foreground">
        {step.title}
      </h3>
      <p className="text-sm leading-relaxed text-muted-foreground">
        {step.description}
      </p>
    </div>
  );
}

function CtaReveal() {
  const { ref, isVisible } = useScrollReveal<HTMLDivElement>({ threshold: 0.3 });

  return (
    <div ref={ref} className={cn(isVisible ? "animate-scale-in" : "opacity-0")}>
      <Link
        href="/translate"
        className="group inline-flex items-center justify-center gap-2 rounded-xl bg-primary px-8 py-4 text-lg font-semibold text-primary-foreground no-underline shadow-lg shadow-primary/25 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-primary/30 active:translate-y-0"
      >
        Try It Now
        <svg
          className="h-5 w-5 transition-transform duration-300 group-hover:translate-x-1"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
        </svg>
      </Link>
    </div>
  );
}

/* ============================
   Page
   ============================ */
export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main>
        <HeroSection />
        <StatsSection />
        <FeaturesSection />
        <HowItWorksSection />
      </main>
      <footer className="border-t border-border bg-card px-6 py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-4 text-center">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="hsl(var(--primary-foreground))" />
              </svg>
            </div>
            <span className="text-lg font-bold text-foreground">SignBridge</span>
          </div>
          <p className="max-w-md text-sm leading-relaxed text-muted-foreground">
            Accessible, real-time sign language translation. Built to empower
            deaf and hearing communities to communicate freely.
          </p>
          <p className="text-xs text-muted-foreground">
            Made with care for accessibility.
          </p>
        </div>
      </footer>
    </div>
  );
}
