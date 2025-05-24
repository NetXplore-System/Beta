import React, { useEffect, useState } from "react";
import { Routes, Route, useLocation, HashRouter } from "react-router-dom";

// Pages
import ResearchWizard from "./pages/ResearchWizard.jsx";
// import Home from "./pages/Home.jsx";
import SignIn from "./pages/SignIn.jsx";
import SignUp from "./pages/SignUp.jsx";
import Profile from "./pages/Profile.jsx";
import EditProfile from "./pages/EditProfile.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import ChoosePlatform from "./pages/ChoosePlatform.jsx";
import Welcome from "./pages/Welcome.jsx";

// Components
import Header from "./components/Header/Header.jsx";
import Menu from "./components/Menu/Menu.jsx";
import PrivateRoute from "./components/PrivateRoute.jsx";
// import HomeW from "./pages/Home/HomeWikipedia.jsx";

// State
import { useDispatch, useSelector } from "react-redux";
import { setUser } from "./redux/user/userSlice";

// UI
import { Toaster } from "sonner";
import { Tooltip } from 'react-tooltip';



function AppContent() {
  const [isOpen, setIsOpen] = useState(false);
  const [isAppReady, setIsAppReady] = useState(false);
  const { currentUser } = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const location = useLocation();

  useEffect(() => {
    const savedUser = localStorage.getItem("user");
    const savedToken = localStorage.getItem("token");

    if (savedUser && savedToken) {
      dispatch(
        setUser({
          user: JSON.parse(savedUser),
          access_token: savedToken,
        })
      );
    }
    setIsAppReady(true);
  }, [dispatch]);

  if (!isAppReady) return null;

  const isWelcomePage = location.pathname === "/" && !currentUser;
  const isAuthPage =
    location.pathname === "/signin" ||
    location.pathname === "/register" ||
    isWelcomePage;

  return (
    <>
      <Tooltip id="my-tooltip" place="top" type="dark" effect="solid" />
      <Toaster position="top-center" />


      {!isAuthPage && (
        <>
          <Menu isOpen={isOpen} setIsOpen={setIsOpen} />
          <Header isOpen={isOpen} />
        </>
      )}

      <div className={`main-content ${isOpen ? "expanded" : "collapsed"}`}>
        <Routes>
          {/* Public Routes */}
          <Route path="/signin" element={<SignIn />} />
          <Route path="/register" element={<SignUp />} />
          <Route path="/choose-platform" element={<ChoosePlatform />} />
          {/* <Route path="/explore" element={<Home />} /> */}
          <Route path="/newresearch" element={<ResearchWizard />} />
          {/* <Route path="/home_wikipedia" element={<HomeW />} /> */}

          <Route path="/" element={currentUser ? <Dashboard /> : <Welcome />} />

          {/* Protected Routes */}
          <Route element={<PrivateRoute />}>
            <Route path="/profile" element={<Profile />} />
            <Route path="/edit-profile" element={<EditProfile />} />
            <Route path="/history" element={<History />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Route>
        </Routes>
      </div>
    </>
  );
}

function App() {
  return (
    <HashRouter>
      <AppContent />
    </HashRouter>
  );
}

export default App;
