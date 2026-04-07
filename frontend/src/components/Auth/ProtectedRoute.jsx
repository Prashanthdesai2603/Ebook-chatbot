import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  const isAuthenticated = token && token !== 'undefined' && token !== 'null';

  if (!isAuthenticated) {
    // Redirect to root if not authenticated
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ProtectedRoute;
