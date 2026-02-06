"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export function Navbar() {
  const pathname = usePathname();

  return (
    <header role="banner" className="sticky top-0 z-50">
      <nav
        aria-label="Main navigation"
        className="flex items-center justify-between border-b border-border bg-card/80 px-6 py-4 backdrop-blur-md"
      >
        <Link
          href="/"
          className="flex items-center gap-3 text-foreground no-underline"
          aria-label="SignBridge home"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary transition-transform duration-300 hover:scale-110 hover:-rotate-6" aria-hidden="true">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                fill="hsl(var(--primary-foreground))"
              />
            </svg>
          </div>
          <span className="text-xl font-bold tracking-tight text-foreground">
            SignBridge
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <Link
            href="/"
            className={cn(
              "rounded-lg px-4 py-2.5 text-sm font-medium transition-colors no-underline",
              pathname === "/"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground",
            )}
          >
            Home
          </Link>
          <Link
            href="/translate"
            className={cn(
              "rounded-lg px-4 py-2.5 text-sm font-medium transition-colors no-underline",
              pathname === "/translate"
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground",
            )}
          >
            Translate
          </Link>
        </div>
      </nav>
    </header>
  );
}
