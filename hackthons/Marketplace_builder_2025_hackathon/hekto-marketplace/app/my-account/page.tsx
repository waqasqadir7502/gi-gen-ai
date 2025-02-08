import "./account.css"
import Image from "next/image";
import Link from "next/link";
import PageName from "../components/Page name bar/pageName";

export default function MyAccount() {
  return (
    <div>
       <PageName name="My Account"/>
      <div className="account-sec flex flex-col justify-center items-center">
        <div className="account-inner ">
          <form className="sign-in-form flex flex-col items-center justify-center gap-y-4">
            <h3>Log In</h3>
            <p>Please login using account detail bellow.</p>
            <input type="email"  placeholder="Email Address"/>
            <input type="password"   placeholder="Password"/>
            <p><Link href="">Forgot your password?</Link></p>
            <button>Sign In</button>
            <p> Dont have an Account? <Link href="">Create Account</Link></p>
          </form>
        </div>
      </div>
      <div className="brand-banner flex justify-center items-center">
        <Image
          src="/homepage/partnered-firms.png"
          alt=""
          width={904}
          height={93}
        />
      </div>
    </div>
  );
}
