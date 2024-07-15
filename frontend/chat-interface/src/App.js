import { ChakraProvider } from '@chakra-ui/react';
import Chat from './components/Chat';

function App() {
  return (
    <ChakraProvider>
      <div className="App">
        <Chat />
      </div>
    </ChakraProvider>
  );
}

export default App;