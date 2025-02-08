import Image from "next/image";
import Link from "next/link";
import "./navbar.css";

export default function Navbar() {
  return (
    <div>
      <div className="top-section flex justify-center">
        <div className="inner-section flex justify-between ">
          <div className="top-col-1 flex gap-4">
            <p className="flex items-center gap-2">
              <Image
                src="./icons/uil_envelope-alt.svg"
                alt="envelope"
                width={20}
                height={20}
              />
              mhhasanul@gmail.com
            </p>
            <p className="flex items-center gap-2 ml-6">
              <Image
                src="./icons/bx_bx-phone-call.svg"
                alt="phone icon"
                width={20}
                height={20}
              />
              (12345)67890
            </p>
          </div>
          <div className="top-col-2 flex gap-4 items-center">
            <p className="flex items-center gap-1">
              English
              <Image
                src="./icons/dropdown-icon.svg"
                alt="dropdown icon"
                width={20}
                height={20}
              />
            </p>
            <p className="flex items-center gap-1">
              USD
              <Image
                src="./icons/dropdown-icon.svg"
                alt="dropdown icon"
                width={20}
                height={20}
              />
            </p>
            <p className="flex items-top gap-1">
              Login
              <Image
                src="./icons/user.svg"
                alt="user icon"
                width={20}
                height={20}
              />
            </p>
            <p className="flex items-center gap-1">
              Wishlist
              <Image
                src="./icons/heart.svg"
                alt="heart icon"
                width={20}
                height={20}
              />
            </p>
            <p className="flex items-center gap-2 ml-6">
              <Image
                src="./icons/cart.svg"
                alt="cart icon"
                width={20}
                height={20}
              />
            </p>
          </div>
        </div>
      </div>
      <div className="bottom-section flex justify-center items-center">
        <div className="inner-section flex justify-between">
          <div className="bottom-col-1 flex gap-6 ">
            <h1 className="logo">Hekto</h1>
            <nav className="flex items-center">
              <ul className="flex gap-4 ml-6">
                <li>
                  <Link href="/" className="homebtn flex items-center gap-1 ">
                    Home
                    <Image
                      src="./icons/pink-caret.svg"
                      alt="dropdown icon"
                      width={20}
                      height={20}
                    />
                  </Link>
                </li>
                <li>Pages</li>
                <li>
                  <Link href="/products">Products</Link>
                </li>
                <li>
                  <Link href="/blog">Blog</Link>
                </li>
                <li>
                  <Link href="/about">About Us</Link>
                </li>
                <li>
                  <Link href="/shoplist">Shop</Link>
                </li>
                <li>
                  <Link href="/contactus">Contact</Link>
                </li>
              </ul>
            </nav>
          </div>
          <div className="bottom-col-2 flex justify-center">
            <input type="text" className="searchinp" />
            <button className="searchbtn flex items-center justify-center">
              <Image
                src="./icons/search.svg"
                alt="searchIcon"
                width={20}
                height={20}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
