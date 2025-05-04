import { useState } from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import "./App.css";
import Home from "./pages/Home.jsx";
import SignIn from "./pages/SignIn.jsx";
import SignUp from "./pages/SignUp.jsx";
import Profile from "./pages/Profile.jsx";
import EditProfile from "./pages/EditProfile.jsx";
import PrivateRoute from "./components/PrivateRoute";
import Header from "./components/Header/Header.jsx";
import Menu from "./components/Menu/Menu.jsx";
// import HomeW from "./pages/HomeW.jsx";
import HomeW from "./pages/HomeWikipedia.jsx";
import History from "./pages/History.jsx";
import ChoosePlatform from "./pages/ChoosePlatform.jsx";
import { Toaster } from "react-hot-toast";
import { useSelector } from "react-redux";
import { HashRouter } from "react-router-dom";


function AppContent() {
  const [isOpen, setIsOpen] = useState(false);
  const { currentUser } = useSelector((state) => state.user);
  const location = useLocation();
  
  const isAuthPage = location.pathname === '/sign-in' || location.pathname === '/' || location.pathname === '/sign-up';
  
  return (
    <>
      <Toaster
        position="top-center"
        reverseOrder={false}
        gutter={8}
        toastOptions={{
          className: '',
          duration: 5000,
          removeDelay: 1000,
          style: {
            background: '#050d2d',
            color: '#fff',
            fontWeight: 'bold',
            fontSize: '12px',
            marginTop: '30px',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: 'green',
              secondary: 'black',
            },  
          },
        }}
      />
      
      {!isAuthPage && (
        <>
          <Menu isOpen={isOpen} setIsOpen={setIsOpen} />
          <Header isOpen={isOpen} />
        </>
      )}
      
      <div className={`main-content ${isOpen ? "expanded" : "collapsed"}`}>
        <Routes>
          <Route path="/" element={<SignIn />} />
          <Route path="/home" element={<Home />} />
          <Route path="/home_wikipedia" element={<HomeW />} />
          <Route path="/sign-in" element={<SignIn />} />
          <Route path="/sign-up" element={<SignUp />} />
          <Route path="/choose-platform" element={<ChoosePlatform />} />
          <Route element={<PrivateRoute />}>
            <Route path="/profile" element={<Profile />} />
            <Route path="/edit-profile" element={<EditProfile />} />
            <Route path="/history" element={<History />} />
          </Route>
        </Routes>
      </div>
    </>
  );
}

function App() {
  return (
    // <BrowserRouter>
    <HashRouter>
      <AppContent />
    </HashRouter>
    // </BrowserRouter>
  );
}

export default App;