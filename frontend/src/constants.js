// frontend/src/constants.js

export const WELCOME = `
👋 **Hi, I'm _SpillSense Assistant_.**

**What I can do for you:**

**Upload a CSV**  
- I'll automatically run the saved model, append a **predicted_severity** column, and give you a downloadable file.

**Upload an image (jpg/png)**  
- I'll tell you if it likely contains a spill or not.

**plot & visualize**  
- Type **plot <something>** (e.g., \`plot correlation\`, \`plot histogram quantity\`).  
- Only one plot per request; you'll get the image right in the chat.

**train**  
- Retrain the model on the latest processed CSVs.

**predict severity <csv_path>**  
- Manually trigger prediction on a CSV already on the server (e.g., in \`uploads/\`).

**analyze hsi**  
- Run the hyperspectral (.mat) workflow and download the PDF report.

**scrape news <keywords> <YYYY-MM-DD YYYY-MM-DD>**  
- Build a news PDF about a topic (e.g., oil spills).

**help**  
- Show this list again.
`;
