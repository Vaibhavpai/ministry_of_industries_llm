document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const sendBtn = document.getElementById('sendBtn');
    
    let currentBaseContext = "";

    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'system-message'} fade-in`;
        
        if (typeof content === 'string') {
            const innerDiv = document.createElement('div');
            innerDiv.className = 'message-content';
            innerDiv.innerHTML = `<p>${content}</p>`;
            messageDiv.appendChild(innerDiv);
        } else {
            messageDiv.appendChild(content);
        }
        
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    }

    function showTypingIndicator() {
        const template = document.getElementById('typingTemplate');
        const indicator = template.content.cloneNode(true);
        const wrapper = document.createElement('div');
        wrapper.id = 'typingIndicatorWrapper';
        wrapper.className = 'message system-message';
        wrapper.appendChild(indicator);
        chatContainer.appendChild(wrapper);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const wrapper = document.getElementById('typingIndicatorWrapper');
        if (wrapper) wrapper.remove();
    }

    function renderResultCard(result) {
        // Add conversational text first
        addMessage("Based on your description, here’s the most relevant classification:");

        const template = document.getElementById('resultCardTemplate');
        const cardWrapper = document.createElement('div');
        cardWrapper.className = 'message system-message';
        
        const card = template.content.cloneNode(true);
        
        card.querySelector('.nic-code').textContent = `NIC ${result.top_nics[0].nic_code}`;
        card.querySelector('.sector-label').textContent = result.division;
        card.querySelector('.activity-label').textContent = result.top_nics[0].nic_label;
        
        // Confidence formatting
        const confPercent = Math.round(result.top_nics[0].confidence * 100);
        card.querySelector('.confidence-text').textContent = `${confPercent}%`;
        
        const barFill = card.querySelector('.confidence-bar-fill');
        const textLabel = card.querySelector('.confidence-text-label');
        
        // Slight delay for animation
        setTimeout(() => {
            barFill.style.width = `${confPercent}%`;
            if (confPercent >= 80) {
                barFill.style.backgroundColor = 'var(--conf-high)';
                textLabel.textContent = `${confPercent}% • High Confidence`;
            } else if (confPercent >= 50) {
                barFill.style.backgroundColor = 'var(--conf-med)';
                textLabel.textContent = `${confPercent}% • Medium Confidence`;
            } else {
                barFill.style.backgroundColor = 'var(--conf-low)';
                textLabel.textContent = `${confPercent}% • Low Confidence`;
            }
        }, 100);

        // Explanation
        card.querySelector('.explanation-text').textContent = result.explanation;
        
        // Keywords
        const keywordsUl = card.querySelector('.keywords-ul');
        if (result.keywords && result.keywords.length > 0) {
            result.keywords.forEach(kw => {
                const li = document.createElement('li');
                li.textContent = kw;
                keywordsUl.appendChild(li);
            });
        } else {
            card.querySelector('.keywords-list').style.display = 'none';
        }

        // Guidance
        const g = result.guidance;
        card.querySelector('.portal-text').textContent = g.portal || '—';
        card.querySelector('.registration-text').textContent = g.registration || '—';
        card.querySelector('.compliance-text').textContent = g.compliance || '—';
        
        const schemesContainer = card.querySelector('.schemes-tags');
        if (g.schemes && g.schemes.length > 0) {
            g.schemes.forEach(s => {
                const span = document.createElement('span');
                span.className = 'scheme-tag';
                span.textContent = s;
                schemesContainer.appendChild(span);
            });
        } else {
            schemesContainer.textContent = '—';
        }
        
        cardWrapper.appendChild(card);
        chatContainer.appendChild(cardWrapper);
        scrollToBottom();
    }

    function renderClarification(clarification) {
        addMessage("I found a few possibilities. To give you the exact code, could you clarify:");

        const template = document.getElementById('optionsTemplate');
        const wrapper = document.createElement('div');
        wrapper.className = 'message system-message';
        
        const container = template.content.cloneNode(true);
        container.querySelector('.clarification-question').textContent = clarification.question;
        
        const optionsDiv = container.querySelector('.buttons-grid');
        
        if (clarification.options && clarification.options.length > 0) {
            clarification.options.forEach(opt => {
                const btn = document.createElement('button');
                btn.className = 'option-btn';
                btn.textContent = opt;
                btn.onclick = () => handleClarificationSelect(opt, wrapper);
                optionsDiv.appendChild(btn);
            });
        } else {
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'option-btn';
            input.style.cursor = 'text';
            input.placeholder = 'Type here and press Enter...';
            input.onkeypress = (e) => {
                if (e.key === 'Enter') {
                    handleClarificationSelect(input.value, wrapper);
                }
            };
            optionsDiv.appendChild(input);
        }
        
        wrapper.appendChild(container);
        chatContainer.appendChild(wrapper);
        scrollToBottom();
    }

    async function handleClarificationSelect(answer, optionsWrapperNode) {
        if (!answer.trim()) return;
        
        // Disable buttons so they can't be clicked again
        const buttons = optionsWrapperNode.querySelectorAll('.option-btn');
        buttons.forEach(b => {
            b.disabled = true;
            b.style.opacity = '0.5';
            b.style.cursor = 'default';
        });

        addMessage(answer, true);
        
        const combinedText = `${currentBaseContext} ${answer}`;
        currentBaseContext = ""; // Reset
        
        await fetchPrediction(combinedText);
    }

    async function fetchPrediction(text) {
        showTypingIndicator();
        sendBtn.disabled = true;
        
        // Artificial delay to make it feel like AI is "thinking"
        await new Promise(r => setTimeout(r, 600));

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            removeTypingIndicator();
            
            if (data.status === 'clarification_needed') {
                currentBaseContext = text;
                renderClarification(data.clarification);
            } else if (data.status === 'success') {
                renderResultCard(data.result);
            } else {
                addMessage("Sorry, an error occurred while processing your request.", false);
            }
        } catch (error) {
            removeTypingIndicator();
            addMessage("Network error. Make sure the server is running.", false);
            console.error(error);
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;
        
        addMessage(text, true);
        userInput.value = '';
        currentBaseContext = ""; // Reset
        
        fetchPrediction(text);
    });
});
