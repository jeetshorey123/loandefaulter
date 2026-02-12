document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading
        loading.style.display = 'block';
        results.style.display = 'none';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        try {
            // Call API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            
            // Hide loading
            loading.style.display = 'none';
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            console.error('Error:', error);
            loading.style.display = 'none';
            alert('Error making prediction. Please try again.');
        }
    });
    
    function displayResults(result) {
        const predictionBox = document.getElementById('predictionBox');
        const riskTitle = document.getElementById('riskTitle');
        const probabilityText = document.getElementById('probabilityText');
        const gaugeFill = document.getElementById('gaugeFill');
        const gaugeText = document.getElementById('gaugeText');
        const metricsGrid = document.getElementById('metricsGrid');
        const recommendation = document.getElementById('recommendation');
        
        const probability = result.default_probability;
        const isHighRisk = probability >= 50;
        
        // Set risk class
        predictionBox.className = 'prediction-box ' + (isHighRisk ? 'risky' : 'safe');
        
        // Set risk title
        if (probability < 30) {
            riskTitle.textContent = '✅ LOW RISK';
            riskTitle.style.color = '#28a745';
        } else if (probability < 50) {
            riskTitle.textContent = '⚠️ MODERATE RISK';
            riskTitle.style.color = '#ffc107';
        } else if (probability < 70) {
            riskTitle.textContent = '🔴 HIGH RISK';
            riskTitle.style.color = '#dc3545';
        } else {
            riskTitle.textContent = '🚨 VERY HIGH RISK';
            riskTitle.style.color = '#a00';
        }
        
        // Set probability
        probabilityText.textContent = probability.toFixed(2) + '%';
        probabilityText.style.color = isHighRisk ? '#dc3545' : '#28a745';
        
        // Animate gauge
        gaugeFill.className = 'gauge-fill ' + (isHighRisk ? 'high-risk' : '');
        setTimeout(() => {
            gaugeFill.style.height = probability + '%';
        }, 100);
        gaugeText.textContent = probability.toFixed(1) + '%';
        
        // Display metrics
        metricsGrid.innerHTML = '';
        const metrics = result.metrics || {};
        
        for (let [key, value] of Object.entries(metrics)) {
            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card';
            metricCard.innerHTML = `
                <h4>${formatMetricName(key)}</h4>
                <div class="value">${formatMetricValue(value)}</div>
            `;
            metricsGrid.appendChild(metricCard);
        }
        
        // Display recommendations
        let recommendationHTML = '<h3>💡 Recommendations</h3><ul>';
        
        if (isHighRisk) {
            recommendationHTML += `
                <li>Conduct additional credit verification</li>
                <li>Consider requiring a co-signer or additional collateral</li>
                <li>Implement closer monitoring of payment schedule</li>
                <li>Review loan terms and possibly offer a lower amount</li>
                <li>Contact borrower to discuss financial situation</li>
            `;
        } else {
            recommendationHTML += `
                <li>Borrower shows good repayment potential</li>
                <li>Standard monitoring procedures recommended</li>
                <li>Consider this borrower for future loan products</li>
                <li>Maintain regular payment reminders</li>
                <li>Monitor for any changes in payment patterns</li>
            `;
        }
        
        recommendationHTML += '</ul>';
        recommendation.innerHTML = recommendationHTML;
        
        // Show results with animation
        results.style.display = 'block';
        results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    function formatMetricName(name) {
        return name.replace(/_/g, ' ')
                   .split(' ')
                   .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                   .join(' ');
    }
    
    function formatMetricValue(value) {
        if (typeof value === 'number') {
            if (value % 1 === 0) {
                return value.toString();
            }
            return value.toFixed(2);
        }
        return value;
    }
});
