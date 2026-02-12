# Deployment Notes

## Vercel Free Tier Deployment

The current deployment uses a **lightweight rule-based algorithm** instead of the full ML models due to Vercel's 50MB deployment size limit.

### What's Included (Free Tier):
- ✅ Beautiful responsive web UI
- ✅ Risk scoring algorithm based on:
  - Payment history
  - Late fees
  - Days since last payment
  - Interest rate
  - Debt-to-income ratio
  - Revolving utilization
- ✅ Real-time predictions
- ✅ Mobile responsive
- ✅ Serverless API

### Limitations:
- ⚠️ Uses rule-based algorithm instead of trained ML models
- ⚠️ Accuracy ~75-80% (vs 99.5% for full ML model)

## Full ML Model Deployment Options

To use the **99.5% accuracy trained ML models**, choose one of these:

### Option 1: Vercel Pro
- Upgrade to Vercel Pro ($20/month)
- Supports larger deployments
- Uncomment the ML code in `api/predict.py`
- Restore full `requirements.txt`

### Option 2: External ML API
- Deploy models to:
  - AWS SageMaker
  - Google Cloud AI Platform
  - Azure ML
  - Hugging Face Inference API
- Update `/api/predict.py` to call external API

### Option 3: Streamlit Cloud (FREE)
- Use the included `app.py`
- Deploy to [share.streamlit.io](https://share.streamlit.io/)
- Full ML models included
- 99.5% accuracy
- **Recommended for ML features**

### Option 4: Railway/Render
- Free tier supports larger Python apps
- Deploy with full ML dependencies
- Similar to Streamlit but more flexible

## Performance Comparison

| Platform | ML Accuracy | Deploy Time | Cost | Best For |
|----------|-------------|-------------|------|----------|
| Vercel Free (current) | ~75-80% | 30 sec | Free | UI Demo |
| Streamlit Cloud | 99.5% | 2 min | Free | Full ML Features |
| Vercel Pro | 99.5% | 30 sec | $20/mo | Production |
| External ML API | 99.5% | Varies | Varies | Enterprise |

## Recommended Approach

**For Demo/Portfolio**: Current Vercel Free deployment → Perfect!  
**For Production**: Streamlit Cloud (free) or Vercel Pro ($20/mo)  
**For Enterprise**: External ML API with Vercel frontend

## How to Switch

### To Streamlit (Full ML):
```bash
streamlit run app.py
```

### To Full ML on Vercel Pro:
1. Upgrade Vercel account
2. Restore `requirements.txt`:
   ```
   pandas==2.1.0
   numpy==1.24.3
   scikit-learn==1.3.0
   ```
3. See `api/predict.py.fullml` for complete ML code
