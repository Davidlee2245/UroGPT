# UroGPT UI - Gemini-Style Landing Page

A beautiful, accessible, Gemini-inspired landing interface for UroGPT built with React, TypeScript, Vite, and Tailwind CSS.

## 🚀 Features

- **Responsive Design**: Works perfectly on mobile (360px), tablet (768px), and desktop (1280px+)
- **Keyboard Accessible**: Full keyboard navigation with proper focus management
- **Modern UI**: Clean, minimalist design inspired by Google Gemini
- **UroGPT Branding**: Uses teal color scheme (#10a37f) from UroGPT logo
- **Component-Based**: Reusable, well-documented components

## 📦 Tech Stack

- React 18
- TypeScript
- Vite
- Tailwind CSS
- Lucide React (icons)

## 🛠️ Setup & Installation

### Quick Start

```bash
# Create project (if starting fresh)
npm create vite@latest urogpt-ui -- --template react-ts

# Navigate to project
cd urogpt-ui

# Install dependencies
npm install

# Install dev dependencies
npm install -D tailwindcss postcss autoprefixer

# Initialize Tailwind
npx tailwindcss init -p

# Install Lucide icons
npm install lucide-react

# Start development server
npm run dev
```

### Or Clone This Project

```bash
cd urogpt-ui
npm install
npm run dev
```

The app will be available at `http://localhost:3000`

## 📁 Project Structure

```
urogpt-ui/
├── src/
│   ├── components/
│   │   ├── ActionChips.tsx      # Action buttons row
│   │   ├── Banner.tsx           # Bottom promotional banner
│   │   ├── Chip.tsx            # Reusable chip button
│   │   ├── IconButton.tsx      # Reusable icon button
│   │   ├── MainContent.tsx     # Central content area
│   │   ├── ModelSelector.tsx   # Model dropdown
│   │   ├── RecentItem.tsx      # Sidebar recent chat item
│   │   ├── SearchBar.tsx       # Main search input
│   │   ├── Sidebar.tsx         # Left navigation panel
│   │   └── ToolsMenu.tsx       # Tools popover menu
│   ├── App.tsx                 # Main app component
│   ├── main.tsx               # Entry point
│   └── index.css              # Global styles
├── index.html
├── tailwind.config.ts
├── tsconfig.json
├── vite.config.ts
└── package.json
```

## 🎨 Design Highlights

### Color Scheme
- Primary: `#10a37f` (UroGPT teal)
- Background: White and light gray tones
- Text: Gray scale for hierarchy

### Typography
- Font: Inter, system sans-serif stack
- Clean, professional appearance
- No decorative icons in main UI

### Accessibility Features
- ✅ Keyboard navigation (Tab, Arrow keys, Enter, Escape, /)
- ✅ Focus indicators
- ✅ ARIA labels and roles
- ✅ Skip-to-content link
- ✅ Color contrast ≥ 4.5:1
- ✅ Screen reader friendly

### Responsive Breakpoints
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## ⌨️ Keyboard Shortcuts

- `/` - Focus search bar
- `Escape` - Close menus/modals
- `Arrow keys` - Navigate menus
- `Enter` - Select/submit
- `Tab` - Navigate between elements

## 🎯 Components Overview

### Sidebar
- New chat button
- Explore Gems button
- Recent chats list (15 mock items)
- Activity & Settings at bottom
- Collapsible on mobile

### SearchBar
- Large input field
- Plus button (left)
- Tools menu
- Model selector
- Mic button (right)
- Auto-suggestions on focus

### ActionChips
- Create Image
- Write
- Build
- Deep Research
- Create Video

### Banner
- Dismissible notification
- Call-to-action button
- Fixed at bottom center

## 🔧 Customization

### Colors
Edit `tailwind.config.ts`:

```typescript
colors: {
  'urogpt': {
    500: '#10a37f', // Main teal
    600: '#0d8c6f', // Darker teal
    // ... more shades
  }
}
```

### Recent Items
Edit mock data in `src/components/Sidebar.tsx`:

```typescript
const recentChats = [
  'Your custom item',
  // ... more items
]
```

## 📱 Mobile Behavior

- Sidebar hidden by default
- Hamburger menu button appears
- Overlay when sidebar open
- Touch-friendly button sizes
- Responsive text sizing

## 🚢 Build for Production

```bash
npm run build
```

Outputs to `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 🐛 Known Issues

None currently. Report issues via GitHub.

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Please ensure:
- TypeScript types are correct
- Accessibility is maintained
- Components are documented
- No console errors

## 👨‍💻 Author

Built for UroGPT - AI-Powered Urinalysis Assistant

---

**Enjoy building with UroGPT UI!** 🚀

