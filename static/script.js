// static/script.js
new Vue({
    el: '.container',
    delimiters: ['[[', ']]'],
    data: {
        features: [
            { label: 'Age', value: null },
            { label: 'Sex', value: null },
            { label: 'BMI', value: null },
            { label: 'Blood Pressure', value: null },
            { label: 'S1', value: null },
            { label: 'S2', value: null },
            { label: 'S3', value: null },
            { label: 'S4', value: null },
            { label: 'S5', value: null },
            { label: 'S6', value: null }
        ],
        results: {
            diagnosis: '-',
            probability: 0,
            mse: 0
        }
    },
    methods: {
        validateInputs() {
            return this.features.every(f => 
                f.value !== null && 
                !isNaN(f.value) && 
                typeof f.value === 'number'
            );
        },
        predict() {
            if (!this.validateInputs()) {
                alert('Please fill all fields with valid numbers');
                return;
            }
            this.sendRequest(this.features.map(f => f.value));
        },
        predictWithGD() {
            this.predict(); // For now, same functionality
        },
        sendRequest(features) {
            const cleanedFeatures = features.map(v => 
                v === null || isNaN(v) ? 0 : parseFloat(v)
            );
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ features: cleanedFeatures })
            })
            .then(response => {
                if (!response.ok) throw new Error('Server error');
                return response.json();
            })
            .then(data => {
                if (data.error) throw new Error(data.error);
                this.results = {
                    diagnosis: data.diagnosis || '-',
                    probability: data.probability || 0,
                    mse: data.mse || 0
                };
            })
            .catch(error => {
                console.error('Error:', error);
                alert(`Error: ${error.message}`);
                this.results = { diagnosis: '-', probability: 0, mse: 0 };
            });
        }
    }
});