document.getElementById("send-btn").addEventListener("click", function() {
    var userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    var chatBox = document.getElementById("chat-box");
    var timestamp = new Date().toLocaleTimeString();

    // Add user message to chat box
    chatBox.innerHTML += '<div class="user-message" style="background-color: #e1f5fe; padding: 5px; margin-bottom: 5px; border-radius: 4px;"><strong>You:</strong> ' + userInput + '<br><small>' + timestamp + '</small></div>';

    fetch('/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'message=' + encodeURIComponent(userInput)
    })
    .then(response => response.json())
    .then(data => {
        var aiResponse = data.response;

        // Add AI response to chat box
        chatBox.innerHTML += '<div class="ai-message" style="background-color: #fff9c4; padding: 5px; margin-bottom: 5px; border-radius: 4px;"><strong>AI:</strong> ' + aiResponse + '<br><small>' + timestamp + '</small></div>';

        // Clear input field
        document.getElementById("user-input").value = '';

        // Scroll chat box to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        chatBox.innerHTML += '<div class="error-message" style="color: red; padding: 5px; margin-bottom: 5px;"><strong>Error:</strong> ' + error + '</div>';
    });
});
