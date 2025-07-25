document.getElementById("merchantForm").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form from submitting the default way

    const merchantId = document.getElementById("merchant_id").value;

    const response = await fetch("http://localhost:5000/check_merchant", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ merchant_id: merchantId })
    });

    const result = await response.json();

    // Display the result
    const resultDiv = document.getElementById("result");
    if (response.ok) {
        resultDiv.innerHTML = `<p>Merchant is ${result.valid ? 'valid' : 'invalid'}. ${result.message}</p>`;
    } else {
        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
    }
});
