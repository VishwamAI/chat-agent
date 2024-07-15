// Chat.js
// This file contains the main chat component for the SSI chat functionality.
// It includes a text input field, a submit button, and a chat display area.

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  Input,
  VStack,
  HStack,
  Text,
  useToast
} from '@chakra-ui/react';

const Chat = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSubmit = async () => {
    if (input.trim()) {
      setMessages(prevMessages => [...prevMessages, { sender: 'user', text: input }]);
      setIsLoading(true);
      try {
        const response = await fetch('http://127.0.0.1:5000/generate_response', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ input: input }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setMessages(prevMessages => [...prevMessages, { sender: 'ai', text: data.response }]);
      } catch (error) {
        console.error('Error:', error);
        toast({
          title: 'Error',
          description: `Failed to get response from the server: ${error.message}`,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
        setInput('');
      }
    } else {
      toast({
        title: 'Message is empty',
        description: "Please enter a message before sending.",
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  return (
    <VStack spacing={4}>
      <Box w="100%" p={4} bg="gray.100" maxHeight="400px" overflowY="auto" ref={chatContainerRef}>
        {messages.map((message, index) => (
          <Box key={index} bg={message.sender === 'user' ? 'blue.100' : 'green.100'} p={2} mb={2} borderRadius="md">
            <Text fontWeight={message.sender === 'user' ? 'bold' : 'normal'}>{message.text}</Text>
          </Box>
        ))}
      </Box>
      <HStack w="100%">
        <Input
          placeholder="Type your message here..."
          value={input}
          onChange={handleInputChange}
        />
        <Button colorScheme="blue" onClick={handleSubmit} isLoading={isLoading}>
          Send
        </Button>
      </HStack>
    </VStack>
  );
};

export default Chat;