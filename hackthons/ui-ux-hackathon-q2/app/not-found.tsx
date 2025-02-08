import "./globals.css";
import Image from "next/image";
import PageName from "./components/Page name bar/pageName";

export default function NotFound() {
  return (
    <div>
      <PageName name="404 Not Found" />

      <div className="not-found-sec flex justify-center">
        <div className="not-found-inner flex flex-col items-center justify-center gap-y-10">
          <Image
            src="/404/notfound.png"
            alt="Not Found"
            width={913}
            height={387}
          />
          <h3> oops! The page you requested was not found!</h3>
          <button>Back To Home</button>
        </div>
      </div>
      <div className="brand-banner flex justify-center items-center mt-5 mb-5">
        <Image
          src="/homepage/partnered-firms.png"
          alt=""
          width={1246}
          height={128}
        />
      </div>
    </div>
  );
}
