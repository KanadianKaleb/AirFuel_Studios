# Airfuel Studios

**Professional results. Beginner knowledge.**

Accessible vehicle log analysis and tuning — designed for everyone.  
Effortlessly analyze VE, MAF, and timing tables, get instant heatmaps, and export formats ready for HP Tuners, SCT, MegaSquirt, or your favorite app.

---

## Features

- **Smart log import:** Drop in HP Tuners, EFILive, or generic CSV logs. Auto-detects headers; lets you fix mapping fast.
- **Preset-driven mapping:** HP Tuners (Gen3/Gen4), EFILive, SCT, MegaSquirt.
- **VE, MAF, and timing corrections:** Dynamic binning, heatmaps, and instant tuning advice.  
  *Includes: timing delta calculation, spark advance preview, and export-ready timing adjustment tables.*
- **HP Tuners imports/exports:** Multiple header variants, percent/absolute modes, CRLF/UTF-8 BOM controls.
- **Beginner-friendly UI:** No gatekeeping, full preview before download.
- **Advanced:** SCT/MegaSquirt pivots, sim mode, multi-file comparison, .csv repository, timing analysis built-in.
- **Large file support:** Warnings for files >50MB, flexible pandas options.
- **File switching:** Safe reset and CSV repository for easy swaps.

---

## Quick Start

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
3. **Load data**
   - Drag-and-drop a log file, or use "Load local file" to browse.
   - Use presets for instant mapping, or adjust column mappings as needed.
   - Apply header candidate if initial mapping looks off.

4. **Analyze VE, MAF, and timing**
   - Preview corrections and heatmaps for all tables including spark/timing.
   - Export tables ready to import into HP Tuners/SCT/MegaSquirt as needed.
   - Try the simulator to see AFR/VE/timing impact.
   - Use comparison mode for log-to-log differences.

---

## Deployment to Streamlit Community Cloud

Deploy this app to Streamlit Community Cloud so users can access it without installing anything locally:

1. **Fork or clone this repository** to your GitHub account
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
3. **Click "New app"** and select:
   - Repository: `YourUsername/AirFuel_Studios`
   - Branch: `main` (or your preferred branch)
   - Main file path: `app.py`
4. **Click "Deploy"**

Streamlit Community Cloud will automatically:
- Install dependencies from `requirements.txt`
- Apply configuration from `.streamlit/config.toml`
- Deploy your app with a public URL

**Note:** Users visiting your deployed app won't need to install Streamlit, Python, or any dependencies - everything runs in the cloud!

---

## Example Screenshots

*Add screenshots here (.png or .gif for UI preview: VE, MAF, and timing adjustment visualization).*

---

## License

MIT License (see [LICENSE](LICENSE))

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute features, including new timing adjustment models.

---

## Community and Support

- Discussion: [GitHub Issues](https://github.com/airfuelstudios/airfuelstudios/issues)
- Feature requests: Open an issue (timing features, binning methods, UI).
- Beginner questions welcome!

---

*Accessible tuning for everyone. No paywall, no gatekeeping. Professional results — whatever your skill level, for spark/timing too!*
