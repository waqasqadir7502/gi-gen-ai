import { CountryDetails } from "@/app/types/types";
import Link from "next/link";

const CountryName = ({ params }: { params: { country_name: string } }) => {
  const countryDetails: CountryDetails[] = [
    {
      name: "Pakistan",
      capital: "Islamabad",
      population: "241 million",
    },
    {
      name: "USA",
      capital: "Washington D.C",
      population: "345 Million",
    },
    {
      name: "Korea",
      capital: "Seoul",
      population: "51.7 Million",
    },
    {
      name: "China",
      capital: "Beijing",
      population: "1.409 billion",
    },
    {
      name: "Morocco",
      capital: "Rabat",
      population: "38.1 Million",
    },
    {
      name: "England",
      capital: "London",
      population: "57.7 million",
    },
  ];

  const country= countryDetails.find((c) =>
    c.name.toLowerCase() === params.country_name.toLowerCase());

  if(!country){
    return(
        <h1>Country Doesn't Exist</h1>
    )
  }

  return (
    <div className="flex flex-column w-full justify-center text-2xl mt-40 leading-9">
      <div >
        <h1><b>Country Name: </b>{country.name}</h1>
        <h1><b>Capital : </b>{country.capital}</h1>
        <h1><b>Current Population: </b>{country.population}</h1>
      </div>
        <Link href="/country" className="absolute bottom-5 right-5 shadow-md p-5 hover:text-white hover:bg-black"> {">"}Back To Featured Countries</Link>
    </div>
  );
};

export default CountryName;
