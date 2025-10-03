function sendMessage() {
    let messageInput = document.getElementById('message-input');
    let message = messageInput.value;
    displayMessage('user', message)
    
    // Get the selected function from the dropdown menu
    let functionSelect = document.getElementById('function-select');
    let selectedFunction = functionSelect.value;
    
    // Send an AJAX request to the Flask API endpoint based on the selected function
    let xhr = new XMLHttpRequest();
    let url;

    switch (selectedFunction) {
        case 'search':
            url = '/search';
            break;
        case 'kbanswer':
            url = '/kbanswer';
            break;
        case 'answer':
            url = '/answer';
            break;
        default:
            url = '/answer';
    }
    
    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            displayMessage('assistant', response.message);
        }
    };
    xhr.send(JSON.stringify({message: message}));
    
    // Clear the input field
    messageInput.value = '';
}

function displayMessage(sender, message) {
    let chatContainer = document.getElementById('chat-container');
    let messageDiv = document.createElement('div');

    if (sender === 'assistant') {
        messageDiv.classList.add('assistant-message');
        
        // Create a span for the Chatbot text
        let chatbotSpan = document.createElement('span');
        chatbotSpan.innerHTML = "<b>Chatbot:</b> ";
        messageDiv.appendChild(chatbotSpan);
        
        // Append the message to the Chatbot span
        messageDiv.innerHTML += message;
    } else {
        messageDiv.classList.add('user-message');

        let userSpan = document.createElement('span');
        userSpan.innerHTML = "<b>User:</b> ";
        messageDiv.appendChild(userSpan);
        
        // Append the message to the span
        messageDiv.innerHTML += message;
    }

    // Create a timestamp element
    let timestamp = document.createElement('span');
    timestamp.classList.add('timestamp');
    let currentTime = new Date().toLocaleTimeString();
    timestamp.innerText = " ["+ currentTime+"]";
    messageDiv.appendChild(timestamp);

    chatContainer.appendChild(messageDiv);

    // Scroll to the bottom of the chat container
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle button click event
var sendButton = document.getElementById('send-btn');
sendButton.addEventListener('click', sendMessage);

// Handle Enter key press event
var messageInput = document.getElementById('message-input');
messageInput.addEventListener('keypress', function(event) {
    if (event.key === 13) {
        event.preventDefault();
        sendMessage();
    }
});