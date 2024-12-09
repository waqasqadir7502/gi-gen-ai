import type { Metadata } from "next";
import localFont from "next/font/local";
import {Josefin_Sans} from "next/font/google"
import {Lato} from "next/font/google"
import "./globals.css";
import Navbar from "./components/Navbar/page";
import Footer from "./components/footer/page";

export const lato = Lato({
  variable : "--font-lato",
  subsets: ['latin'],
  display: 'swap',
  weight : "400" ,
});

export const josefinSans = Josefin_Sans({
  variable : "--font-josefin-sans",
  subsets: ['latin'],
  display: 'swap',
  weight:["100", "300","400","500","700"]
});
 
export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${josefinSans} ${lato}antialiased`}
      >
        <Navbar/>
        {children}
        <Footer/>
      </body>
    </html>
  );
}