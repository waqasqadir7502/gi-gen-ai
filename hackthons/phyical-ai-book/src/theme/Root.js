import React, { useState, useEffect } from 'react';
import ChatWidget from '../components/ChatWidget/ChatWidget';

// Create a wrapper component that will be used by Docusaurus
const Root = ({ children }) => {
  const [isClient, setIsClient] = useState(false);

  // Ensure we only render on the client side to avoid SSR issues
  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <>
      {children}
      {isClient && <ChatWidget />}
    </>
  );
};

export default Root;