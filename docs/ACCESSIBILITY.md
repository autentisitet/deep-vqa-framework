# Frontend accessibility checks

The frontend includes responsive layout, keyboard file selection, visible focus
styles, live status regions, reduced-motion support, semantic landmarks, and
accessible names for generated media previews.

Before release, run both automated and manual checks:

1. Open `frontend/index.html` through the deployed site and run Lighthouse's
   Accessibility audit in browser DevTools.
2. Run axe DevTools or another WCAG 2.2 AA scanner against the deployed page.
3. With the mouse disconnected, use `Tab`, `Enter`, `Space`, and `Escape` to
   reach the language switcher, file picker, action buttons, and result regions.
4. Test at 200% zoom and at narrow mobile widths.
5. Enable reduced motion and a screen reader, then verify service, evaluation,
   error, and visualization status announcements.

Automated tools are useful gates, but passing them is not a formal WCAG
certification. A formal claim still requires human review and representative
assistive-technology testing.
