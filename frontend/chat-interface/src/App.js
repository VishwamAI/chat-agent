import { ChakraProvider } from '@chakra-ui/react';
import Chat from './components/Chat';

function App() {
  return (
    <ChakraProvider>
      <Chat />
    </ChakraProvider>
  );
}

export default App;