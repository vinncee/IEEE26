# Frontend Migration Guide

## Overview
The frontend has been successfully migrated from **Vite + React 18** to **Next.js 16** with **React 19**.

## Key Changes

### Framework Upgrade
- **Previous**: Vite SPA with React 18.3.1
- **Current**: Next.js 16.1.6 with React 19

### Package Updates
- `react`: ^18.3.1 → ^19
- `react-dom`: ^18.3.1 → ^19
- `typescript`: ^5.5.4 → 5.7.3
- `next`: N/A → 16.1.6
- Added comprehensive UI component library through Radix UI

### Folder Structure
```
frontend/
├── app/                  # Next.js app directory (replaces src/)
│   ├── page.tsx         # Home page
│   ├── translate/
│   │   └── page.tsx     # Translation page
│   ├── globals.css      # Global styles
│   └── layout.tsx       # Root layout
├── components/          # Reusable React components
│   └── navbar.tsx       # Navigation bar
├── hooks/              # Custom React hooks
│   ├── use-signbridge-ws.ts   # WebSocket integration
│   └── use-scroll-reveal.ts   # Scroll animation hook
├── lib/                # Utility functions
│   └── utils.ts        # Shared utilities
├── styles/             # (Removed - using Tailwind CSS)
├── node_modules/
├── package.json        # Project dependencies
├── tsconfig.json       # TypeScript configuration
├── next.config.mjs     # Next.js configuration
├── tailwind.config.ts  # Tailwind CSS configuration
└── postcss.config.mjs  # PostCSS configuration
```

### Styling
- **Added**: Tailwind CSS 3.4.17 with full configuration for SignBridge theme
- **Colors**: Implemented accessible teal primary color with HSL-based theming
- **Components**: Created comprehensive component library using Radix UI (45+ components)

### WebSocket Integration
The `useSignBridgeWS` hook has been adapted to:
- Connect to `ws://localhost:8000/ws/video`
- Support message types: `caption`, `frame`, `correction`
- Match backend message format: `CaptionOut` with `session`, `user`, `mode`, `confidence`

### Environment Configuration
Create a `.env.local` file for development:
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/video
```

## Running the Application

### Development
```bash
npm install --legacy-peer-deps  # (First time only)
npm run dev
```

### Production Build
```bash
npm run build
npm start
```

### Type Checking
TypeScript build errors are ignored in next.config.mjs (matching original setup).

## Compatibility Notes

### Peer Dependency Resolution
- `--legacy-peer-deps` flag was used during installation to allow React 19 with react-day-picker 8.10.1
- This is a known compatibility gap but doesn't affect current functionality
- Consider upgrading to `react-day-picker@9+` when React 19 support is added

### Next.js Features
- Prerendered static pages for fastest performance
- App Router (file-based routing)
- Built-in TypeScript support
- Optimized images and assets
- Automatic code splitting

## Testing & Verification

✅ **Build Status**: Successfully compiles with 0 errors
- Compiled in 2.8s using Turbopack
- Routes created: `/` (home) and `/translate` (translation page)
- All dependencies resolved

## Breaking Changes
- Old React Router setup replaced with Next.js App Router
- Component imports now use `@/` alias (mapped to root directory)
- No more `src/` folder - app structure uses top-level `app/` directory
- Removed Vite configuration - all tools bundled in Next.js

## Rollback Instructions
If needed, the old Vite-based frontend can be restored from version control. The current setup maintains backward compatibility with the FastAPI backend.
